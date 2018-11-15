#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_translate import common_layers  # noqa
from pytorch_translate import rnn
from pytorch_translate.dictionary import TAGS


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
        pool_type="max",
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

        # model parameters
        self.pool_type = pool_type
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
        char_lengths = torch.sum(1 - encoder_padding_mask, dim=0)
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
        if self.pool_type == "max":
            return torch.max(inputs, dim=dim)[0]

        elif self.pool_type == "mean":
            return torch.mean(inputs, dim=dim)

        elif self.pool_type == "logsumexp":
            logsumexp = inputs.exp().mean(dim=dim, keepdim=True).log()
            return logsumexp.squeeze(dim)

        else:
            raise Exception("Invalid pool type: {}".format(self.pool_type))
