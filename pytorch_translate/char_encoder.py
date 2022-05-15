#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_translate import common_layers, rnn, utils  # noqa
from pytorch_translate.data.dictionary import TAGS
from torch.nn.utils.rnn import pack_padded_sequence


def add_args(parser):
    parser.add_argument(
        "--char-embed-dim",
        type=int,
        default=128,
        metavar="N",
        help=("Character embedding dimension."),
    )
    parser.add_argument(
        "--char-rnn-units",
        type=int,
        default=256,
        metavar="N",
        help=("Number of units for Character LSTM."),
    )
    parser.add_argument(
        "--char-rnn-layers",
        type=int,
        default=1,
        metavar="N",
        help=("Number of Character LSTM layers."),
    )
    parser.add_argument(
        "--char-cnn-params",
        type=str,
        metavar="EXPR",
        help=("String experission, [(dim, kernel_size), ...]."),
    )
    parser.add_argument(
        "--char-cnn-nonlinear-fn",
        type=str,
        default="tanh",
        metavar="EXPR",
        help=("Nonlinearity applied to char conv outputs. Values: relu, tanh."),
    )
    parser.add_argument(
        "--char-cnn-num-highway-layers",
        type=int,
        default=0,
        metavar="N",
        help=("Char cnn encoder highway layers."),
    )
    parser.add_argument(
        "--char-cnn-output-dim",
        type=int,
        default=-1,
        metavar="N",
        help="Output dim of the CNN layer. If set to -1, this is computed "
        "from char-cnn-params.",
    )
    parser.add_argument(
        "--use-pretrained-weights",
        type=utils.bool_flag,
        nargs="?",
        const=True,
        default=False,
        help="Use pretrained weights for the character model including "
        "the char embeddings, CNN filters, highway networks",
    )
    parser.add_argument(
        "--finetune-pretrained-weights",
        type=utils.bool_flag,
        nargs="?",
        const=True,
        default=False,
        help="Boolean flag to specify whether or not to update the "
        "pretrained weights as part of training",
    )
    parser.add_argument(
        "--pretrained-weights-file",
        type=str,
        default="",
        help=("Weights file for loading pretrained weights"),
    )
    parser.add_argument(
        "--unk-only-char-encoding",
        type=utils.bool_flag,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Boolean flag. When True, taking words embeddings"
            "for in-vocab tokens and char encoder's outputs for oov tokens"
            "When False, concatenating words embeddings and char encoder's outputs"
            "for all tokens."
        ),
    )


class HighwayLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        transform_activation=F.relu,
        gate_activation=F.softmax,
        # Srivastava et al. (2015) recommend initializing bT to a negative
        # value, in order to militate the initial behavior towards carry.
        # We initialized bT to a small interval around −2
        gate_bias=-2,
    ):
        super().__init__()
        self.highway_transform_activation = transform_activation
        self.highway_gate_activation = gate_activation
        self.highway_transform = nn.Linear(input_dim, input_dim)
        self.highway_gate = nn.Linear(input_dim, input_dim)
        self.highway_gate.bias.data.fill_(gate_bias)

    def forward(self, x):
        transform_output = self.highway_transform_activation(self.highway_transform(x))
        gate_output = self.highway_gate_activation(self.highway_gate(x))

        transformation_part = torch.mul(transform_output, gate_output)
        # TODO: https://github.com/pytorch/pytorch/issues/10747 makes the
        # torch.FloatTensor() expression necessary. Once that gets fixed we
        # can just write 1 - gate_output
        carry_part = torch.mul(
            (torch.FloatTensor([1.0]).type_as(gate_output) - gate_output), x
        )
        return torch.add(transformation_part, carry_part)


class CharCNNModel(nn.Module):
    """
    A Conv network to generate word embedding from character embeddings, from
    Character-Aware Neural Language Models, https://arxiv.org/abs/1508.06615.

    Components include convolutional filters, pooling, and
    optional highway network. We also have the ability to use pretrained ELMo
    which corresponds to the byte embeddings, CNN weights and the highway layer.
    """

    def __init__(
        self,
        dictionary,
        num_chars=50,
        char_embed_dim=32,
        convolutions_params="((128, 3), (128, 5))",
        nonlinear_fn_type="tanh",
        num_highway_layers=0,
        # A value of -1 for char_cnn_output_dim implies no projection layer
        # layer at the output of the highway network
        char_cnn_output_dim=-1,
        use_pretrained_weights=False,
        finetune_pretrained_weights=False,
        weights_file=None,
    ):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.use_pretrained_weights = use_pretrained_weights

        self.convolutions_params = convolutions_params
        self.num_highway_layers = num_highway_layers
        self.char_embed_dim = char_embed_dim
        self.num_embeddings = num_chars
        self.char_cnn_output_dim = char_cnn_output_dim
        self.filter_dims = sum(f[0] for f in self.convolutions_params)

        # If specified, load the pretrained weights from file
        if use_pretrained_weights:
            self._weight_file = weights_file
            self._finetune_pretrained_weights = finetune_pretrained_weights
            self._load_weights()
        else:
            if nonlinear_fn_type == "tanh":
                nonlinear_fn = nn.Tanh
            elif nonlinear_fn_type == "relu":
                nonlinear_fn = nn.ReLU
            else:
                raise Exception("Invalid nonlinear type: {}".format(nonlinear_fn_type))

            self.embed_chars = rnn.Embedding(
                num_embeddings=num_chars,
                embedding_dim=char_embed_dim,
                padding_idx=self.padding_idx,
                freeze_embed=False,
            )
            self.convolutions = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            char_embed_dim,
                            num_filters,
                            kernel_size,
                            padding=kernel_size,
                        ),
                        nonlinear_fn(),
                    )
                    for (num_filters, kernel_size) in self.convolutions_params
                ]
            )

            highway_layers = []
            for _ in range(self.num_highway_layers):
                highway_layers.append(HighwayLayer(self.filter_dims))
            self.highway_layers = nn.ModuleList(highway_layers)

            if char_cnn_output_dim != -1:
                self.projection = nn.Linear(
                    self.filter_dims, self.char_cnn_output_dim, bias=True
                )

    def _load_weights(self):
        """
        Function to load pretrained weights including byte embeddings.
        """
        self.npz_weights = np.load(self._weight_file)
        self._load_byte_embedding()
        self._load_cnn_weights()
        self._load_highway()
        self._load_projection()

    def _load_byte_embedding(self):
        """
        Function to load the pre-trained byte embeddings. We need to ensure that
        the embeddings account for special yoda tags as well.
        """
        char_embed_weights = self.npz_weights["char_embed"]

        num_tags = TAGS.__len__()
        weights = np.zeros(
            (char_embed_weights.shape[0] + num_tags + 1, char_embed_weights.shape[1]),
            dtype="float32",
        )
        weights[1:-num_tags, :] = char_embed_weights

        self.embed_chars = rnn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.char_embed_dim,
            padding_idx=self.padding_idx,
            freeze_embed=self._finetune_pretrained_weights,
        )
        self.embed_chars.weight.data.copy_(torch.FloatTensor(weights))

    def _load_cnn_weights(self):
        """
        Function to load the weights associated with the pretrained CNN filters.
        For this to work correctly, the cnn params specified in the input arguments
        should match up with the pretrained architecture.
        """
        convolutions = []
        for i, (num_filters, kernel_size) in enumerate(self.convolutions_params):
            conv = torch.nn.Conv1d(
                in_channels=self.char_embed_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size,
                bias=True,
            )

            weight = self.npz_weights["W_cnn_{}".format(i)]
            bias = self.npz_weights["b_cnn_{}".format(i)]

            w_reshaped = np.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
            if w_reshaped.shape != tuple(conv.weight.data.shape):
                raise ValueError("Invalid weight file")

            # Load the pretrained weights and bias. In order to ensure the
            # norm of the weights match up with the rest of the model, we need
            # to normalize the pretrained weights. Here we divide by kernel_size
            conv.weight.data.copy_(
                torch.div(torch.FloatTensor(w_reshaped), (kernel_size * 1.0))
            )
            conv.bias.data.copy_(
                torch.div(torch.FloatTensor(bias), (kernel_size * 1.0))
            )

            conv.weight.requires_grad = self._finetune_pretrained_weights
            conv.bias.requires_grad = self._finetune_pretrained_weights

            convolutions.append(nn.Sequential(conv))
        self.convolutions = nn.ModuleList(convolutions)

    def _load_highway(self):
        """
        Function to load the weights associated with the pretrained highway
        network. In order to ensure the norm of the weights match up with the
        rest of the model, we need to normalize the pretrained weights.
        Here we divide by a fixed constant.
        """
        # the highway layers have same dimensionality as the number of cnn filters
        input_dim = sum(f[0] for f in self.convolutions_params)

        highway_layers = []
        for k in range(self.num_highway_layers):
            highway_layer = HighwayLayer(input_dim)

            # Update the transform params
            w_transform = np.transpose(self.npz_weights["W_transform_{}".format(k)])
            b_transform = self.npz_weights["b_transform_{}".format(k)]
            highway_layer.highway_transform.weight.data.copy_(
                torch.div(torch.FloatTensor(w_transform), 6.0)
            )
            highway_layer.highway_transform.bias.data.copy_(
                torch.FloatTensor(b_transform)
            )
            highway_layer.highway_transform.weight.requires_grad = (
                self._finetune_pretrained_weights
            )
            highway_layer.highway_transform.bias.requires_grad = (
                self._finetune_pretrained_weights
            )

            # Update the carry weights
            w_carry = np.transpose(self.npz_weights["W_carry_{}".format(k)])
            highway_layer.highway_gate.weight.data.copy_(
                torch.div(torch.FloatTensor(w_carry), 6.0)
            )
            highway_layer.highway_gate.weight.requires_grad = (
                self._finetune_pretrained_weights
            )
            b_carry = self.npz_weights["b_carry_{}".format(k)]
            highway_layer.highway_gate.bias.data.copy_(torch.FloatTensor(b_carry))
            highway_layer.highway_gate.bias.requires_grad = (
                self._finetune_pretrained_weights
            )

        highway_layers.append(highway_layer)
        self.highway_layers = nn.ModuleList(highway_layers)

    def _load_projection(self):
        """
        Function to load the weights associated with the pretrained projection
        layer. In order to ensure the norm of the weights match up with the
        rest of the model, we need to normalize the pretrained weights.
        Here we divide by a fixed constant.
        """
        input_dim = self.filter_dims

        self.projection = nn.Linear(input_dim, self.char_cnn_output_dim, bias=True)
        weight = self.npz_weights["W_proj"]
        bias = self.npz_weights["b_proj"]
        self.projection.weight.data.copy_(
            torch.div(torch.FloatTensor(np.transpose(weight)), 10.0)
        )
        self.projection.bias.data.copy_(
            torch.div(torch.FloatTensor(np.transpose(bias)), 10.0)
        )

        self.projection.weight.requires_grad = self._finetune_pretrained_weights
        self.projection.bias.requires_grad = self._finetune_pretrained_weights

    def forward(self, char_inds_flat):
        x = self.embed_chars(char_inds_flat)
        encoder_padding_mask = char_inds_flat.eq(self.padding_idx)
        char_lengths = torch.sum(~encoder_padding_mask, dim=0)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        kernel_outputs = []
        for conv in self.convolutions:
            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
            # conv input: [total_words, char_emb_dim, seq_len]
            # conv output: [total_words, in_channel_dim, seq_len]
            conv_output = conv(x.permute(1, 2, 0))
            kernel_outputs.append(conv_output)
        # Pooling over the entire seq
        pools = [self.pooling(conv, char_lengths, dim=2) for conv in kernel_outputs]
        # [total_words, sum(output_channel_dim)]
        encoder_output = torch.cat([p for p in pools], 1)

        for highway_layer in self.highway_layers:
            encoder_output = highway_layer(encoder_output)

        if self.char_cnn_output_dim != -1:
            encoder_output = self.projection(encoder_output)

        # (total_words, output_dim)
        return encoder_output

    def pooling(self, inputs, char_lengths, dim):
        return torch.max(inputs, dim=dim)[0]


class CharRNNModel(nn.Module):
    """Bi-LSTM over characters to produce a word embedding from characters"""

    def __init__(
        self, dictionary, num_chars, char_embed_dim, char_rnn_units, char_rnn_layers
    ):
        super().__init__()
        self.num_chars = num_chars
        self.padding_idx = dictionary.pad()
        self.embed_chars = rnn.Embedding(
            num_embeddings=num_chars,
            embedding_dim=char_embed_dim,
            padding_idx=self.padding_idx,
            freeze_embed=False,
        )

        assert (
            char_rnn_units % 2 == 0
        ), "char_rnn_units must be even (to be divided evenly between directions)"
        self.char_lstm_encoder = rnn.LSTMSequenceEncoder.LSTM(
            char_embed_dim,
            char_rnn_units // 2,
            num_layers=char_rnn_layers,
            bidirectional=True,
        )

        self.onnx_export_model = False

    def forward(self, src_tokens, src_lengths, char_inds, word_lengths):
        # char_inds has shape (batch_size, max_words_per_sent, max_word_len)
        bsz, seqlen, maxchars = char_inds.size()

        if self.onnx_export_model:
            assert bsz == 1
            maxchars_tensor = torch.onnx.operators.shape_as_tensor(char_inds)[2]
            char_inds_flat_shape = torch.cat(
                (torch.LongTensor([-1]), maxchars_tensor.view(1))
            )
            char_inds_flat = torch.onnx.operators.reshape_from_tensor_shape(
                char_inds, char_inds_flat_shape
            ).t()
            char_rnn_input = self.embed_chars(char_inds_flat)
            packed_char_input = pack_padded_sequence(
                char_rnn_input, word_lengths.view(-1)
            )
        else:
            # shape (batch_size, max_words_per_sent)
            nonzero_word_locations = word_lengths > 0

            # (total_words,)
            word_lengths_flat = word_lengths[nonzero_word_locations]

            # (max_word_length, total_words)
            char_inds_flat = char_inds[nonzero_word_locations].t()

            # inputs to RNN must be in descending order of length
            sorted_word_lengths, word_length_order = torch.sort(
                word_lengths_flat, descending=True
            )

            char_rnn_input = self.embed_chars(char_inds_flat[:, word_length_order])

            packed_char_input = pack_padded_sequence(
                char_rnn_input, sorted_word_lengths
            )

        # h_last shape: (num_layers * num_directions, batch_size, hidden_dim)
        _, (h_last, _) = self.char_lstm_encoder(packed_char_input)

        # take last-layer output only (shape: (total_words, hidden_dim))
        # concatenating forward and backward outputs at end/beginning of words
        char_rnn_output = torch.cat((h_last[-2, :, :], h_last[-1, :, :]), dim=1)

        if self.onnx_export_model:
            # (seqlen, bsz==1, char_rnn_units)
            x = char_rnn_output.unsqueeze(1)
        else:
            # "unsort" (total_words, char_rnn_units)
            _, inverted_word_length_order = torch.sort(word_length_order)
            unsorted_rnn_output = char_rnn_output[inverted_word_length_order, :]

            x = char_rnn_output.new(bsz, seqlen, unsorted_rnn_output.shape[1])
            x[nonzero_word_locations] = unsorted_rnn_output
            x = x.transpose(0, 1)  # (seqlen, bsz, char_rnn_units)

        return x

    def prepare_for_onnx_export_(self, **kwargs):
        self.onnx_export_model = True
