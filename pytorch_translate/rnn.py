#!/usr/bin/env python3

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
import torch.onnx.operators
from pytorch_translate import rnn_cell  # noqa

from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.models import (
    FairseqEncoder,
    FairseqModel,
    register_model,
    register_model_architecture,
)

from pytorch_translate import vocab_reduction
from pytorch_translate import word_dropout
from pytorch_translate.ngram import NGramDecoder
from pytorch_translate.multi_model import MultiEncoder, MultiDecoder
from pytorch_translate.common_layers import (
    Embedding,
    RNNLayer,
    Linear,
    DecoderWithOutputProjection,
)
from pytorch_translate import attention


def torch_find(index, query, vocab_size):
    """
    Finds elements of query from index, outputting the last (max) index for each
    query.
    preconditions:  (1) index and query are flat arrays (can be different sizes)
                    (2) all tokens in index and query have values < vocab_size
    """
    full_to_index = (torch.zeros(vocab_size)).long()
    if torch.cuda.is_available():
        full_to_index = full_to_index.cuda()
    index_shape_range = torch.arange(index.shape[0]).long()
    if torch.cuda.is_available():
        index_shape_range = index_shape_range.cuda()
    full_to_index[index] = index_shape_range
    result = full_to_index[query]
    return result


@register_model("rnn")
class RNNModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--dropout",
            default=0.1,
            type=float,
            metavar="D",
            help="dropout probability",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-freeze-embed",
            default=False,
            action="store_true",
            help=(
                "whether to freeze the encoder embedding or allow it to be "
                "updated during training"
            ),
        )
        parser.add_argument(
            "--encoder-hidden-dim", type=int, metavar="N", help="encoder cell num units"
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="number of encoder layers"
        )
        parser.add_argument(
            "--encoder-bidirectional",
            action="store_true",
            help="whether the first layer is bidirectional or not",
        )
        parser.add_argument(
            "--averaging-encoder",
            default=False,
            action="store_true",
            help=(
                "whether use mean encoder hidden states as decoder initial "
                "states or not"
            ),
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-freeze-embed",
            default=False,
            action="store_true",
            help=(
                "whether to freeze the encoder embedding or allow it to be "
                "updated during training"
            ),
        )
        parser.add_argument(
            "--decoder-hidden-dim", type=int, metavar="N", help="decoder cell num units"
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="number of decoder layers"
        )
        parser.add_argument(
            "--decoder-out-embed-dim",
            type=int,
            metavar="N",
            help="decoder output embedding dimension",
        )
        parser.add_argument(
            "--attention-type",
            type=str,
            metavar="EXPR",
            help="decoder attention, defaults to dot",
        )
        parser.add_argument(
            "--residual-level",
            default=None,
            type=int,
            help=(
                "First layer where to apply a residual connection. "
                "The value should be greater than 0 and smaller than the number of "
                "layers."
            ),
        )
        parser.add_argument(
            "--cell-type",
            default="lstm",
            type=str,
            metavar="EXPR",
            help="cell type, defaults to lstm, values:lstm, milstm, layer_norm_lstm",
        )

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument(
            "--encoder-dropout-in",
            type=float,
            metavar="D",
            help="dropout probability for encoder input embedding",
        )
        parser.add_argument(
            "--encoder-dropout-out",
            type=float,
            metavar="D",
            help="dropout probability for encoder output",
        )
        parser.add_argument(
            "--decoder-dropout-in",
            type=float,
            metavar="D",
            help="dropout probability for decoder input embedding",
        )
        parser.add_argument(
            "--decoder-dropout-out",
            type=float,
            metavar="D",
            help="dropout probability for decoder output",
        )
        parser.add_argument(
            "--sequence-lstm",
            action="store_true",
            help="use nn.LSTM implementation for encoder",
        )
        parser.add_argument(
            "--ngram-decoder",
            default=None,
            type=int,
            nargs="+",
            help=(
                "A single integer, or a list of integers. If "
                "positive, the decoder is not recurrent but a feedforward "
                "network with target-side n-gram history as input. The decoder "
                "is still conditioned on the source side via attention. If "
                "this parameter is a list of integers, the n-th entry applies "
                "to the n-th decoder (for multilingual models and "
                "multi-decoders)"
            ),
        )
        parser.add_argument(
            "--ngram-activation-type",
            default="relu",
            type=str,
            metavar="EXPR",
            help=(
                "Activation in FF layers of the ngram decoder, defaults to "
                "relu, values: relu, tanh"
            ),
        )
        parser.add_argument(
            "--multi-encoder",
            default=None,
            type=int,
            help=(
                "If this is positive, train n encoder networks rather than "
                "only one. The outputs of the encoders are concatenated before "
                "passing them through to the decoder."
            ),
        )
        parser.add_argument(
            "--multi-decoder",
            default=None,
            type=int,
            help=(
                "If this is positive, train n decoder networks rather than "
                "only one. The predictions are combined via the method in "
                "--multi-decoder-combination-strategy."
            ),
        )
        parser.add_argument(
            "--multi-decoder-combination-strategy",
            default="bottleneck",
            type=str,
            metavar="EXPR",
            help=(
                "Only used if --multi-decoder is positive. Controls how the "
                "decoders are combined with each other.\n"
                "- uniform: Separate projection layers, average predictions\n"
                "- unprojected: Shared projection layer, unprojected "
                "decoder outputs are averaged.\n"
                "- weighted: Separate projection layers, weighted average "
                "of logits. Weights are learned from unprojected decoder "
                "outputs.\n"
                "- weighted-unprojected: Shared projection layer, weighted "
                "average of decoder outputs. Weights are learned from "
                "unprojected decoder outputs.\n"
                "- concat: Shared projection layer, decoder outputs are "
                "concatenated.\n"
                "- bottleneck: Like 'concat' but with an additional "
                "bottleneck layer to reduce the size of the output embedding "
                "matrix.\n"
                "- multiplicative-unprojected: Shared projection layer, element"
                "-wise product of decoder outputs after ReLU.\n"
            ),
        )

        # Args for vocab reduction
        vocab_reduction.add_args(parser)
        # Args for word dropout
        word_dropout.add_args(parser)

    @staticmethod
    def build_encoder(args, src_dict):
        if args.sequence_lstm:
            encoder_class = LSTMSequenceEncoder
        else:
            encoder_class = RNNEncoder
        return encoder_class(
            src_dict,
            embed_dim=args.encoder_embed_dim,
            freeze_embed=args.encoder_freeze_embed,
            cell_type=args.cell_type,
            num_layers=args.encoder_layers,
            hidden_dim=args.encoder_hidden_dim,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            residual_level=args.residual_level,
            bidirectional=bool(args.encoder_bidirectional),
            word_dropout_params=args.word_dropout_params,
        )

    @staticmethod
    def build_decoder(
        args, src_dict, dst_dict, ngram_decoder=None, project_output=True
    ):
        if ngram_decoder:
            if args.ngram_activation_type == "relu":
                activation_fn = nn.ReLU
            elif args.ngram_activation_type == "tanh":
                activation_fn = nn.Tanh
            else:
                raise Exception(
                    "ngram_activation_type '%s' not implemented"
                    % args.ngram_activation_type
                )
            decoder = NGramDecoder(
                src_dict=src_dict,
                dst_dict=dst_dict,
                n=ngram_decoder,
                encoder_hidden_dim=args.encoder_hidden_dim,
                embed_dim=args.decoder_embed_dim,
                freeze_embed=args.decoder_freeze_embed,
                out_embed_dim=args.decoder_out_embed_dim,
                num_layers=args.decoder_layers,
                hidden_dim=args.decoder_hidden_dim,
                attention_type=args.attention_type,
                dropout_in=args.decoder_dropout_in,
                dropout_out=args.decoder_dropout_out,
                residual_level=args.residual_level,
                activation_fn=activation_fn,
                project_output=project_output,
            )
        else:
            decoder = RNNDecoder(
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
                project_output=project_output,
            )
        return decoder

    @classmethod
    def build_model(cls, args, src_dict, dst_dict):
        """Build a new model instance."""
        base_architecture(args)
        if args.multi_encoder is not None:
            encoders = [
                RNNModel.build_encoder(args, src_dict)
                for _ in range(args.multi_encoder)
            ]
            encoder = MultiEncoder(src_dict, encoders)
        else:
            encoder = RNNModel.build_encoder(args, src_dict)

        if args.multi_decoder is not None:
            ngram_decoder_args = [None] * args.multi_decoder
            if args.ngram_decoder is not None:
                ngram_decoder_args = args.ngram_decoder
                if len(ngram_decoder_args) == 1:
                    ngram_decoder_args = [ngram_decoder_args[0]] * args.multi_decoder
                assert len(ngram_decoder_args) == args.multi_decoder
            decoders = [
                RNNModel.build_decoder(
                    args, src_dict, dst_dict, n, project_output=False
                )
                for n in ngram_decoder_args
            ]
            decoder = MultiDecoder(
                src_dict,
                dst_dict,
                decoders=decoders,
                combination_strategy=args.multi_decoder_combination_strategy,
                split_encoder=args.multi_encoder is not None,
                vocab_reduction_params=args.vocab_reduction_params,
            )
        else:
            if args.multi_encoder:
                args.encoder_hidden_dim *= args.multi_encoder
            n = int(args.ngram_decoder) if args.ngram_decoder else None
            decoder = RNNModel.build_decoder(args, src_dict, dst_dict, n)
        return cls(encoder, decoder)

    def get_targets(self, sample, net_output):
        targets = sample["target"].view(-1)
        possible_translation_tokens = net_output[-1]
        if possible_translation_tokens is not None:
            targets = torch_find(
                possible_translation_tokens, targets, len(self.dst_dict)
            )
        return targets

    @staticmethod
    def expand_encoder_output(encoder_out, n):
        """
        Expand all outputs to replicate each instance from batch in place n
        times (as for beam search)
        """
        (
            unpacked_output,
            final_hiddens,
            final_cells,
            src_lengths,
            src_tokens,
        ) = encoder_out
        unpacked_output = (
            unpacked_output.unsqueeze(2)
            .repeat(1, 1, n, 1)
            .view(unpacked_output.shape[0], -1, unpacked_output.shape[2])
        )
        final_hiddens = (
            final_hiddens.unsqueeze(2)
            .repeat(1, 1, n, 1)
            .view(final_hiddens.shape[0], -1, final_hiddens.shape[2])
        )
        final_cells = (
            final_cells.unsqueeze(2)
            .repeat(1, 1, n, 1)
            .view(final_cells.shape[0], -1, final_cells.shape[2])
        )
        src_lengths = src_lengths.unsqueeze(1).repeat(1, n).view(-1)
        src_tokens = (
            src_tokens.unsqueeze(1).repeat(1, n, 1).view(-1, src_tokens.shape[1])
        )

        return (unpacked_output, final_hiddens, final_cells, src_lengths, src_tokens)


class LSTMSequenceEncoder(FairseqEncoder):
    """RNN encoder using nn.LSTM for cuDNN support / ONNX exportability."""

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
        embed_dim=512,
        freeze_embed=False,
        cell_type="lstm",
        hidden_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        residual_level=None,
        bidirectional=False,
        word_dropout_params=None,
        padding_value=0,
    ):
        assert cell_type == "lstm", 'sequence-lstm requires cell_type="lstm"'

        super().__init__(dictionary)
        self.dictionary = dictionary
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.residual_level = residual_level
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.padding_value = padding_value

        self.embed_tokens = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
            freeze_embed=freeze_embed,
        )
        self.word_dim = embed_dim

        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            is_layer_bidirectional = self.bidirectional and layer == 0
            self.layers.append(
                LSTMSequenceEncoder.LSTM(
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

    def forward(self, src_tokens, src_lengths):
        if LanguagePairDataset.LEFT_PAD_SOURCE:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens, self.padding_idx, left_to_right=True
            )
        if self.word_dropout_module is not None:
            src_tokens = self.word_dropout_module(src_tokens)

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Allows compatibility with Caffe2 inputs for tracing (int32)
        # as well as the current format of Fairseq-Py inputs (int64)
        if src_lengths.dtype is torch.int64:
            src_lengths = src_lengths.int()

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
        unpacked_output, _ = pad_packed_sequence(
            packed_input, padding_value=self.padding_value
        )

        return (unpacked_output, final_hiddens, final_cells, src_lengths, src_tokens)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class RNNEncoder(FairseqEncoder):
    """RNN encoder."""

    def __init__(
        self,
        dictionary,
        word_dropout_params=None,
        embed_dim=512,
        freeze_embed=False,
        hidden_dim=512,
        num_layers=1,
        cell_type="lstm",
        dropout_in=0.1,
        dropout_out=0.1,
        residual_level=None,
        bidirectional=False,
        pretrained_embed=None,
        padding_value=0,
    ):
        super().__init__(dictionary)
        self.dictionary = dictionary
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.residual_level = residual_level
        self.hidden_dim = hidden_dim
        self.output_units = hidden_dim  # fairseq LSTM compatibility
        self.bidirectional = bidirectional
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.padding_value = padding_value

        if pretrained_embed is None:
            self.embed_tokens = Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embed_dim,
                padding_idx=self.padding_idx,
                freeze_embed=freeze_embed,
            )
        else:
            self.embed_tokens = pretrained_embed
        self.word_dim = embed_dim

        self.cell_type = cell_type
        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            self.layers.append(
                RNNLayer(
                    self.word_dim if layer == 0 else hidden_dim,
                    hidden_dim,
                    self.cell_type,
                    True if bidirectional and layer == 0 else False,
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

    def forward(self, src_tokens, src_lengths):
        if LanguagePairDataset.LEFT_PAD_SOURCE:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens, self.padding_idx, left_to_right=True
            )
        if self.word_dropout_module is not None:
            src_tokens = self.word_dropout_module(src_tokens)
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Generate packed seq to deal with varying source seq length
        packed_input, batch_sizes = pack_padded_sequence(x, src_lengths)
        final_hiddens, final_cells = [], []
        next_hiddens = []
        for i, rnn_layer in enumerate(self.layers):
            current_hidden_size = (
                self.hidden_dim // 2 if rnn_layer.is_bidirectional else self.hidden_dim
            )

            if self.cell_type in ["lstm", "milstm", "layer_norm_lstm"]:
                prev_hidden = (
                    x.new(bsz, current_hidden_size).zero_(),
                    x.new(bsz, current_hidden_size).zero_(),
                )
            else:
                raise Exception(f"{self.cell_type} not implemented")

            hidden, current_output = rnn_layer.forward(
                packed_input, prev_hidden, batch_sizes
            )
            next_hiddens.append(hidden)
            prev_hidden = next_hiddens[-1]

            if self.dropout_out != 0:
                current_output = F.dropout(
                    current_output, p=self.dropout_out, training=self.training
                )

            if self.residual_level is not None and i >= self.residual_level:
                packed_input = packed_input.clone() + current_output
            else:
                packed_input = current_output

        final_hiddens, final_cells = zip(*next_hiddens)
        # Reshape to [num_layer, batch_size, hidden_dim]
        final_hiddens = torch.cat(final_hiddens, dim=0).view(
            self.num_layers, *final_hiddens[0].size()
        )
        final_cells = torch.cat(final_cells, dim=0).view(
            self.num_layers, *final_cells[0].size()
        )

        #  [max_seqlen, batch_size, hidden_dim]
        unpacked_output, _ = pad_packed_sequence(
            PackedSequence(packed_input, batch_sizes), padding_value=self.padding_value
        )

        return (unpacked_output, final_hiddens, final_cells, src_lengths, src_tokens)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class RNNDecoder(DecoderWithOutputProjection):
    """RNN decoder."""

    def __init__(
        self,
        src_dict,
        dst_dict,
        vocab_reduction_params=None,
        encoder_hidden_dim=512,
        embed_dim=512,
        freeze_embed=False,
        hidden_dim=512,
        out_embed_dim=512,
        cell_type="lstm",
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        attention_type="dot",
        residual_level=None,
        averaging_encoder=False,
        project_output=True,
    ):
        super().__init__(
            src_dict,
            dst_dict,
            vocab_reduction_params,
            out_embed_dim,
            project_output=project_output,
        )
        self.encoder_hidden_dim = encoder_hidden_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_embed_dim = out_embed_dim
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.attention_type = attention_type
        self.residual_level = residual_level

        num_embeddings = len(dst_dict)
        padding_idx = dst_dict.pad()
        self.embed_tokens = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
            freeze_embed=freeze_embed,
        )
        self.hidden_dim = hidden_dim
        self.averaging_encoder = averaging_encoder

        if cell_type == "lstm":
            cell_class = rnn_cell.LSTMCell
        elif cell_type == "milstm":
            cell_class = rnn_cell.MILSTMCell
        elif cell_type == "layer_norm_lstm":
            cell_class = rnn_cell.LayerNormLSTMCell

        if hidden_dim != encoder_hidden_dim:
            hidden_init_fc_list = []
            cell_init_fc_list = []
            for _ in range(num_layers):
                hidden_init_fc_list.append(Linear(encoder_hidden_dim, hidden_dim))
                cell_init_fc_list.append(Linear(encoder_hidden_dim, hidden_dim))
            self.hidden_init_fc_list = nn.ModuleList(hidden_init_fc_list)
            self.cell_init_fc_list = nn.ModuleList(cell_init_fc_list)
        self.initial_attn_context = nn.Parameter(
            torch.Tensor(encoder_hidden_dim).zero_()
        )

        if attention_type is not None:
            self.attention = attention.build_attention(
                attention_type=attention_type,
                decoder_hidden_state_dim=hidden_dim,
                encoder_output_dim=encoder_hidden_dim,
            )
            self.combined_output_and_context_dim = encoder_hidden_dim + hidden_dim
        else:
            self.attention = None
            self.combined_output_and_context_dim = hidden_dim

        layers = []
        for layer in range(num_layers):
            if layer == 0:
                if self.attention is not None:
                    cell_input_dim = encoder_hidden_dim + embed_dim
                else:
                    cell_input_dim = embed_dim
            else:
                cell_input_dim = hidden_dim
            layers.append(cell_class(input_dim=cell_input_dim, hidden_dim=hidden_dim))
        self.layers = nn.ModuleList(layers)

        if self.combined_output_and_context_dim != out_embed_dim:
            self.additional_fc = Linear(
                self.combined_output_and_context_dim, out_embed_dim
            )

    def forward_unprojected(self, input_tokens, encoder_out, incremental_state=None):
        if incremental_state is not None:
            input_tokens = input_tokens[:, -1:]
        bsz, seqlen = input_tokens.size()

        # get outputs from encoder
        (encoder_outs, final_hidden, final_cell, src_lengths, src_tokens) = encoder_out

        # embed tokens
        x = self.embed_tokens(input_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(
            self, incremental_state, "cached_state"
        )
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            # first time step, initialize previous states
            prev_hiddens, prev_cells = self._init_prev_states(encoder_out)
            input_feed = self.initial_attn_context.expand(bsz, self.encoder_hidden_dim)

        attn_scores_per_step = []
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if self.attention is not None:
                step_input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                step_input = x[j, :, :]
            previous_layer_input = step_input
            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(step_input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                layer_output = F.dropout(
                    hidden, p=self.dropout_out, training=self.training
                )

                if self.residual_level is not None and i >= self.residual_level:
                    # TODO add an assert related to sizes here
                    step_input = layer_output + previous_layer_input
                else:
                    step_input = layer_output
                previous_layer_input = step_input

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            if self.attention is not None:
                out, step_attn_scores = self.attention(
                    hidden, encoder_outs, src_lengths
                )
                input_feed = out
            else:
                combined_output_and_context = hidden
                step_attn_scores = Variable(
                    torch.ones(src_lengths.shape[0], src_lengths.max()).type_as(
                        encoder_outs
                    ),
                    requires_grad=False,
                ).t()
            attn_scores_per_step.append(step_attn_scores.unsqueeze(1))
            attn_scores = torch.cat(attn_scores_per_step, dim=1)
            # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
            attn_scores = attn_scores.transpose(0, 2)
            combined_output_and_context = torch.cat((hidden, out), dim=1)

            # save final output
            outs.append(combined_output_and_context)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self,
            incremental_state,
            "cached_state",
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(
            seqlen, bsz, self.combined_output_and_context_dim
        )

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # bottleneck layer
        if hasattr(self, "additional_fc"):
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
        return x, attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        cached_state = utils.get_incremental_state(
            self, incremental_state, "cached_state"
        )
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, "cached_state", new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def _init_prev_states(self, encoder_out):
        (
            encoder_output,
            final_hiddens,
            final_cells,
            src_lengths,
            src_tokens,
        ) = encoder_out
        num_layers = len(self.layers)
        if self.averaging_encoder:
            # Use mean encoder hidden states
            prev_hiddens = [torch.mean(encoder_output, 0)] * num_layers
        else:
            # Simply return the final state of each layer
            prev_hiddens = [final_hiddens[i] for i in range(num_layers)]
        prev_cells = [final_cells[i] for i in range(num_layers)]

        if hasattr(self, "hidden_init_fc_list"):
            for i in range(num_layers):
                prev_hiddens[i] = self.hidden_init_fc_list[i](prev_hiddens[i])
                prev_cells[i] = self.cell_init_fc_list[i](prev_cells[i])

        return prev_hiddens, prev_cells


@register_model_architecture("rnn", "rnn")
def base_architecture(args):
    # default architecture
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 1)
    args.encoder_hidden_dim = getattr(args, "encoder_hidden_dim", 512)
    args.encoder_bidirectional = getattr(args, "encoder_bidirectional", False)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", args.dropout)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", args.dropout)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_hidden_dim = getattr(args, "decoder_hidden_dim", 512)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.attention_type = getattr(args, "attention_type", "dot")
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.averaging_encoder = getattr(args, "averaging_encoder", False)
    args.encoder_freeze_embed = getattr(args, "encoder_freeze_embed", False)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.ngram_decoder = getattr(args, "ngram_decoder", None)
    args.multi_encoder = getattr(args, "multi_encoder", None)
    args.multi_decoder = getattr(args, "multi_decoder", None)
    args.cell_type = getattr(args, "cell_type", "lstm")
    args.ngram_activation_type = getattr(args, "ngram_activation_type", "relu")
    vocab_reduction.set_arg_defaults(args)
    word_dropout.set_arg_defaults(args)
    args.sequence_lstm = getattr(args, "sequence_lstm", False)
    args.add_encodercoder_output_as_decoder_input = getattr(
        args, "add_encoder_output_as_decoder_input", False
    )


@register_model_architecture("rnn", "rnn_big_test")
def rnn_big_test(args):
    base_architecture(args)
    args.encoder_embed_dim = 1024
    args.encoder_layers = 6
    args.encoder_hidden_dim = 1024
    args.decoder_embed_dim = 1024
    args.decoder_layers = 6
    args.decoder_hidden_dim = 1024
    args.decoder_out_embed_dim = 1024
