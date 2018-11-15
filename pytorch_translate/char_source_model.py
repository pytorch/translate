#!/usr/bin/env python3

import logging
from ast import literal_eval

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import FairseqEncoder, register_model, register_model_architecture
from pytorch_translate import (
    char_encoder,
    model_constants,
    rnn,
    utils,
    vocab_constants,
    word_dropout,
)
from pytorch_translate.common_layers import VariableTracker
from pytorch_translate.dictionary import TAGS
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


logger = logging.getLogger(__name__)


@register_model("char_source")
class CharSourceModel(rnn.RNNModel):
    def __init__(self, task, encoder, decoder):
        super().__init__(task, encoder, decoder)

    @staticmethod
    def add_args(parser):
        rnn.RNNModel.add_args(parser)
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
            "--char-cnn-pool-type",
            type=str,
            default="max",
            metavar="EXPR",
            help=(
                "Pooling function of input sequence outputs. "
                "Values: logsumexp, max, mean, meanmax."
            ),
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

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        src_dict, dst_dict = task.source_dictionary, task.target_dictionary
        base_architecture(args)

        assert args.sequence_lstm, "CharRNNModel only supports sequence_lstm"
        assert args.cell_type == "lstm", "CharRNNModel only supports cell_type lstm"

        assert hasattr(args, "char_source_dict_size"), (
            "args.char_source_dict_size required. "
            "should be set by load_binarized_dataset()"
        )

        if hasattr(args, "char_cnn_params"):
            args.embed_bytes = getattr(args, "embed_bytes", False)

            # If we embed bytes then the number of indices is fixed and does not
            # depend on the dictionary
            if args.embed_bytes:
                num_chars = vocab_constants.NUM_BYTE_INDICES + TAGS.__len__() + 1
            else:
                num_chars = args.char_source_dict_size

            # In case use_pretrained_weights is true, verify the model params
            # are correctly set
            if args.embed_bytes and getattr(args, "use_pretrained_weights", False):
                verify_pretrain_params(args)
            encoder = CharCNNEncoder(
                src_dict,
                num_chars=num_chars,
                embed_dim=args.char_embed_dim,
                token_embed_dim=args.encoder_embed_dim,
                freeze_embed=args.encoder_freeze_embed,
                char_cnn_params=args.char_cnn_params,
                char_cnn_nonlinear_fn=args.char_cnn_nonlinear_fn,
                char_cnn_pool_type=args.char_cnn_pool_type,
                char_cnn_num_highway_layers=args.char_cnn_num_highway_layers,
                char_cnn_output_dim=getattr(args, "char_cnn_output_dim", -1),
                num_layers=args.encoder_layers,
                hidden_dim=args.encoder_hidden_dim,
                dropout_in=args.encoder_dropout_in,
                dropout_out=args.encoder_dropout_out,
                residual_level=args.residual_level,
                bidirectional=bool(args.encoder_bidirectional),
                word_dropout_params=args.word_dropout_params,
                use_pretrained_weights=getattr(args, "use_pretrained_weights", False),
                finetune_pretrained_weights=getattr(
                    args, "finetune_pretrained_weights", False
                ),
                weights_file=getattr(args, "pretrained_weights_file", ""),
            )
        else:
            encoder = CharRNNEncoder(
                src_dict,
                num_chars=args.char_source_dict_size,
                char_embed_dim=args.char_embed_dim,
                token_embed_dim=args.encoder_embed_dim,
                freeze_embed=args.encoder_freeze_embed,
                char_rnn_units=args.char_rnn_units,
                char_rnn_layers=args.char_rnn_layers,
                num_layers=args.encoder_layers,
                hidden_dim=args.encoder_hidden_dim,
                dropout_in=args.encoder_dropout_in,
                dropout_out=args.encoder_dropout_out,
                residual_level=args.residual_level,
                bidirectional=bool(args.encoder_bidirectional),
                word_dropout_params=args.word_dropout_params,
            )

        decoder = rnn.RNNDecoder(
            src_dict=src_dict,
            dst_dict=dst_dict,
            vocab_reduction_params=args.vocab_reduction_params,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            freeze_embed=args.decoder_freeze_embed,
            out_embed_dim=args.decoder_out_embed_dim,
            cell_type=args.cell_type,
            num_layers=args.decoder_layers,
            hidden_dim=args.decoder_hidden_dim,
            attention_type=args.attention_type,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            residual_level=args.residual_level,
            averaging_encoder=args.averaging_encoder,
        )
        return cls(task, encoder, decoder)

    def forward(
        self, src_tokens, src_lengths, char_inds, word_lengths, prev_output_tokens
    ):
        """
        Overriding FairseqModel.forward() due to different encoder inputs.
        """
        encoder_out = self.encoder(src_tokens, src_lengths, char_inds, word_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out


class CharRNNEncoder(FairseqEncoder):
    """
    RNN encoder encoding each word via a bidirectional LSTM over character
    embeddings to obtain word representations, and then an LSTM (optionally
    bidirectional in first layer) to combine word representationsself.

    Uses nn.LSTM for cuDNN support / ONNX exportability.
    """

    @staticmethod
    def LSTM(input_size, hidden_size, **kwargs):
        m = nn.LSTM(input_size, hidden_size, **kwargs)
        for name, param in m.named_parameters():
            if "weight" in name or "bias" in name:
                param.data.uniform_(-0.1, 0.1)
        return m

    def __init__(
        self,
        dictionary,
        num_chars,
        char_embed_dim,
        token_embed_dim,
        freeze_embed=False,
        char_rnn_units=256,
        char_rnn_layers=1,
        hidden_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        residual_level=None,
        bidirectional=False,
        word_dropout_params=None,
    ):

        super().__init__(dictionary)
        self.dictionary = dictionary
        self.num_chars = num_chars
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.residual_level = residual_level
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        num_tokens = len(dictionary)
        self.padding_idx = dictionary.pad()

        self.embed_chars = rnn.Embedding(
            num_embeddings=num_chars,
            embedding_dim=char_embed_dim,
            padding_idx=self.padding_idx,
            freeze_embed=freeze_embed,
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

        self.embed_tokens = None
        if token_embed_dim > 0:
            self.embed_tokens = rnn.Embedding(
                num_embeddings=num_tokens,
                embedding_dim=token_embed_dim,
                padding_idx=self.padding_idx,
                freeze_embed=freeze_embed,
            )

        self.word_dim = char_rnn_units + token_embed_dim

        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            is_layer_bidirectional = self.bidirectional and layer == 0
            if is_layer_bidirectional:
                assert hidden_dim % 2 == 0, (
                    "encoder_hidden_dim must be even if encoder_bidirectional "
                    "(to be divided evenly between directions)"
                )
            self.layers.append(
                rnn.LSTMSequenceEncoder.LSTM(
                    self.word_dim if layer == 0 else hidden_dim,
                    hidden_dim // 2 if is_layer_bidirectional else hidden_dim,
                    num_layers=1,
                    dropout=self.dropout_out,
                    bidirectional=is_layer_bidirectional,
                )
            )

        self.num_layers = len(self.layers)
        self.word_dropout_module = None
        if (
            word_dropout_params
            and word_dropout_params["word_dropout_freq_threshold"] is not None
            and word_dropout_params["word_dropout_freq_threshold"] > 0
        ):
            self.word_dropout_module = word_dropout.WordDropout(
                dictionary, word_dropout_params
            )

        # disables sorting and word-length thresholding if True
        # (enables ONNX tracing of length-sorted input with batch_size = 1)
        self.onnx_export_model = False

    def prepare_for_onnx_export_(self, **kwargs):
        self.onnx_export_model = True

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

        if self.embed_tokens is not None:
            embedded_tokens = self.embed_tokens(src_tokens)

            # (seqlen, bsz, token_embed_dim)
            embedded_tokens = embedded_tokens.transpose(0, 1)

            # (seqlen, bsz, total_word_embed_dim)
            x = torch.cat([x, embedded_tokens], dim=2)

        if self.dropout_in != 0:
            x = F.dropout(x, p=self.dropout_in, training=self.training)

        embedded_words = x

        # Generate packed seq to deal with varying source seq length
        # packed_input is of type PackedSequence, which consists of:
        # element [0]: a tensor, the packed data, and
        # element [1]: a list of integers, the batch size for each step
        packed_input = pack_padded_sequence(x, src_lengths)

        final_hiddens, final_cells = [], []
        for i, rnn_layer in enumerate(self.layers):
            if self.bidirectional and i == 0:
                h0 = x.new(2, bsz, self.hidden_dim // 2).zero_()
                c0 = x.new(2, bsz, self.hidden_dim // 2).zero_()
            else:
                h0 = x.new(1, bsz, self.hidden_dim).zero_()
                c0 = x.new(1, bsz, self.hidden_dim).zero_()

            # apply LSTM along entire sequence
            current_output, (h_last, c_last) = rnn_layer(packed_input, (h0, c0))

            # final state shapes: (bsz, hidden_dim)
            if self.bidirectional and i == 0:
                # concatenate last states for forward and backward LSTM
                h_last = torch.cat((h_last[0, :, :], h_last[1, :, :]), dim=1)
                c_last = torch.cat((c_last[0, :, :], c_last[1, :, :]), dim=1)
            else:
                h_last = h_last.squeeze(dim=0)
                c_last = c_last.squeeze(dim=0)

            final_hiddens.append(h_last)
            final_cells.append(c_last)

            if self.residual_level is not None and i >= self.residual_level:
                packed_input[0] = packed_input.clone()[0] + current_output[0]
            else:
                packed_input = current_output

        # Reshape to [num_layer, batch_size, hidden_dim]
        final_hiddens = torch.cat(final_hiddens, dim=0).view(
            self.num_layers, *final_hiddens[0].size()
        )
        final_cells = torch.cat(final_cells, dim=0).view(
            self.num_layers, *final_cells[0].size()
        )

        #  [max_seqlen, batch_size, hidden_dim]
        unpacked_output, _ = pad_packed_sequence(packed_input)

        return (
            unpacked_output,
            final_hiddens,
            final_cells,
            src_lengths,
            src_tokens,
            embedded_words,
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder all outputs according to new_order."""
        return rnn.reorder_encoder_output(encoder_out, new_order)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class CharCNNEncoder(FairseqEncoder):
    """
    Character-level CNN encoder to generate word representations, as input to
    RNN encoder.
    """

    def __init__(
        self,
        dictionary,
        num_chars=50,
        embed_dim=32,
        token_embed_dim=256,
        freeze_embed=False,
        char_cnn_params="[(128, 3), (128, 5)]",
        char_cnn_nonlinear_fn="tanh",
        char_cnn_pool_type="max",
        char_cnn_num_highway_layers=0,
        char_cnn_output_dim=-1,
        hidden_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        residual_level=None,
        bidirectional=False,
        word_dropout_params=None,
        use_pretrained_weights=False,
        finetune_pretrained_weights=False,
        weights_file=None,
    ):

        super().__init__(dictionary)
        self.dictionary = dictionary
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.residual_level = residual_level
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        convolutions_params = literal_eval(char_cnn_params)
        self.char_cnn_encoder = char_encoder.CharCNNModel(
            dictionary,
            num_chars,
            embed_dim,
            convolutions_params,
            char_cnn_nonlinear_fn,
            char_cnn_pool_type,
            char_cnn_num_highway_layers,
            char_cnn_output_dim,
            use_pretrained_weights,
            finetune_pretrained_weights,
            weights_file,
        )

        self.embed_tokens = None
        num_tokens = len(dictionary)
        self.padding_idx = dictionary.pad()
        if token_embed_dim > 0:
            self.embed_tokens = rnn.Embedding(
                num_embeddings=num_tokens,
                embedding_dim=token_embed_dim,
                padding_idx=self.padding_idx,
                freeze_embed=freeze_embed,
            )
        self.word_dim = (
            char_cnn_output_dim
            if char_cnn_output_dim != -1
            else sum(out_dim for (out_dim, _) in convolutions_params)
        )
        self.word_dim = self.word_dim + token_embed_dim

        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            is_layer_bidirectional = self.bidirectional and layer == 0
            if is_layer_bidirectional:
                assert hidden_dim % 2 == 0, (
                    "encoder_hidden_dim must be even if encoder_bidirectional "
                    "(to be divided evenly between directions)"
                )
            self.layers.append(
                rnn.LSTMSequenceEncoder.LSTM(
                    self.word_dim if layer == 0 else hidden_dim,
                    hidden_dim // 2 if is_layer_bidirectional else hidden_dim,
                    num_layers=1,
                    dropout=self.dropout_out,
                    bidirectional=is_layer_bidirectional,
                )
            )

        self.num_layers = len(self.layers)
        self.word_dropout_module = None
        if (
            word_dropout_params
            and word_dropout_params["word_dropout_freq_threshold"] is not None
            and word_dropout_params["word_dropout_freq_threshold"] > 0
        ):
            self.word_dropout_module = word_dropout.WordDropout(
                dictionary, word_dropout_params
            )

        # Variable tracker
        self.tracker = VariableTracker()
        # Initialize adversarial mode
        self.set_gradient_tracking_mode(False)

    def set_gradient_tracking_mode(self, mode=True):
        """ This allows AdversarialTrainer to turn on retrain_grad when
        running adversarial example generation model."""
        self.tracker.reset()
        self.track_gradients = mode

    def forward(self, src_tokens, src_lengths, char_inds, word_lengths):
        self.tracker.reset()
        # char_inds has shape (batch_size, max_words_per_sent, max_word_len)
        bsz, seqlen, maxchars = char_inds.size()
        # char_cnn_encoder takes input (max_word_length, total_words)
        char_inds_flat = char_inds.view(-1, maxchars).t()
        # output (total_words, encoder_dim)
        char_cnn_output = self.char_cnn_encoder(char_inds_flat)
        x = char_cnn_output.view(bsz, seqlen, char_cnn_output.shape[-1])
        x = x.transpose(0, 1)  # (seqlen, bsz, char_cnn_output_dim)

        if self.embed_tokens is not None:
            embedded_tokens = self.embed_tokens(src_tokens)
            # (seqlen, bsz, token_embed_dim)
            embedded_tokens = embedded_tokens.transpose(0, 1)
            # (seqlen, bsz, total_word_embed_dim)
            x = torch.cat([x, embedded_tokens], dim=2)

        self.tracker.track(x, "token_embeddings", retain_grad=self.track_gradients)
        if self.dropout_in != 0:
            x = F.dropout(x, p=self.dropout_in, training=self.training)
        embedded_words = x

        # The rest is the same as CharRNNEncoder, so could be refactored
        # Generate packed seq to deal with varying source seq length
        # packed_input is of type PackedSequence, which consists of:
        # element [0]: a tensor, the packed data, and
        # element [1]: a list of integers, the batch size for each step
        packed_input = pack_padded_sequence(x, src_lengths)

        final_hiddens, final_cells = [], []
        for i, rnn_layer in enumerate(self.layers):
            if self.bidirectional and i == 0:
                h0 = x.data.new(2, bsz, self.hidden_dim // 2).zero_()
                c0 = x.data.new(2, bsz, self.hidden_dim // 2).zero_()
            else:
                h0 = x.data.new(1, bsz, self.hidden_dim).zero_()
                c0 = x.data.new(1, bsz, self.hidden_dim).zero_()

            # apply LSTM along entire sequence
            current_output, (h_last, c_last) = rnn_layer(packed_input, (h0, c0))

            # final state shapes: (bsz, hidden_dim)
            if self.bidirectional and i == 0:
                # concatenate last states for forward and backward LSTM
                h_last = torch.cat((h_last[0, :, :], h_last[1, :, :]), dim=1)
                c_last = torch.cat((c_last[0, :, :], c_last[1, :, :]), dim=1)
            else:
                h_last = h_last.squeeze(dim=0)
                c_last = c_last.squeeze(dim=0)

            final_hiddens.append(h_last)
            final_cells.append(c_last)

            if self.residual_level is not None and i >= self.residual_level:
                packed_input[0] = packed_input.clone()[0] + current_output[0]
            else:
                packed_input = current_output

        # Reshape to [num_layer, batch_size, hidden_dim]
        final_hiddens = torch.cat(final_hiddens, dim=0).view(
            self.num_layers, *final_hiddens[0].size()
        )
        final_cells = torch.cat(final_cells, dim=0).view(
            self.num_layers, *final_cells[0].size()
        )

        #  [max_seqlen, batch_size, hidden_dim]
        unpacked_output, _ = pad_packed_sequence(packed_input)

        return (
            unpacked_output,
            final_hiddens,
            final_cells,
            src_lengths,
            src_tokens,
            embedded_words,
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder all outputs according to new_order."""
        return rnn.reorder_encoder_output(encoder_out, new_order)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


def verify_pretrain_params(args):
    """
    Function which verifies the model params in case we are using a pretrained
    model like ELMo. This is needed because the model params must match up
    with the weights that are being loaded.
    """
    assert (
        args.embed_bytes
    ), "To use pretrained weights, embed_bytes must be set to True."

    assert (
        args.char_cnn_nonlinear_fn == model_constants.PRETRAINED_CHAR_CNN_NONLINEAR_FN
    ), "To use pretrained weights, the non linearity used should be relu."

    assert (
        args.char_embed_dim == model_constants.PRETRAINED_CHAR_EMBED_DIM
    ), "To use pretrained weights char_embed_dim should be set to 16."

    assert (
        args.char_cnn_output_dim == model_constants.PRETRAINED_CHAR_CNN_OUTPUT_DIM
    ), "To use pretrained weights, the output dim of the CNN layer should be 512."
    assert (
        literal_eval(args.char_cnn_params) == model_constants.PRETRAINED_CHAR_CNN_PARAMS
    ), "CNN Params don't match with the ones needed for loading pretrained weights"


@register_model_architecture("char_source", "char_source")
def base_architecture(args):
    # default architecture
    rnn.base_architecture(args)
    args.char_rnn_units = getattr(args, "char_embed_dim", 128)
    args.char_rnn_units = getattr(args, "char_rnn_units", 256)
    args.char_rnn_layers = getattr(args, "char_rnn_layers", 1)
