#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    PackedSequence,
    pad_packed_sequence,
)
import torch.onnx.operators


from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqModel,
    register_model,
    register_model_architecture,
)
from fbtranslate import rnn_cell  # noqa
from fbtranslate import vocab_reduction
from fbtranslate import word_dropout


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


@register_model('rnn')
class RNNModel(FairseqModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--dropout',
            default=0.1,
            type=float,
            metavar='D',
            help='dropout probability',
        )
        parser.add_argument(
            '--encoder-embed-dim',
            type=int,
            metavar='N',
            help='encoder embedding dimension',
        )
        parser.add_argument(
            '--encoder-freeze-embed',
            default=False,
            action='store_true',
            help=('whether to freeze the encoder embedding or allow it to be '
                  'updated during training'),
        )
        parser.add_argument(
            '--encoder-hidden-dim',
            type=int,
            metavar='N',
            help='encoder cell num units',
        )
        parser.add_argument(
            '--encoder-layers',
            type=int,
            metavar='N',
            help='number of encoder layers',
        )
        parser.add_argument(
            '--encoder-bidirectional',
            action='store_true',
            help='whether the first layer is bidirectional or not',
        )
        parser.add_argument(
            '--averaging-encoder',
            default=False,
            action='store_true',
            help=(
                'whether use mean encoder hidden states as decoder initial '
                'states or not'
            ),
        )
        parser.add_argument(
            '--add-encoder-outputs-as-decoder-input',
            default=False,
            action='store_true',
            help=(
                'whether use max encoder hidden states as constant decoder '
                'input'
            ),
        )
        parser.add_argument(
            '--decoder-embed-dim',
            type=int,
            metavar='N',
            help='decoder embedding dimension',
        )
        parser.add_argument(
            '--decoder-freeze-embed',
            default=False,
            action='store_true',
            help=('whether to freeze the encoder embedding or allow it to be '
                  'updated during training'),
        )
        parser.add_argument(
            '--decoder-hidden-dim',
            type=int,
            metavar='N',
            help='decoder cell num units',
        )
        parser.add_argument(
            '--decoder-layers',
            type=int,
            metavar='N',
            help='number of decoder layers',
        )
        parser.add_argument(
            '--decoder-out-embed-dim',
            type=int,
            metavar='N',
            help='decoder output embedding dimension',
        )
        parser.add_argument(
            '--attention-type',
            type=str,
            metavar='EXPR',
            help='decoder attention, defaults to dot',
        )
        parser.add_argument(
            '--residual-level',
            default=None,
            type=int,
            help=(
                'First layer where to apply a residual connection. '
                'The value should be greater than 0 and smaller than the number of '
                'layers.'
            ),
        )
        parser.add_argument(
            '--cell-type',
            default='lstm',
            type=str,
            metavar='EXPR',
            help='cell type, defaults to lstm, values:lstm, milstm, layer_norm_lstm',
        )

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument(
            '--encoder-dropout-in',
            type=float,
            metavar='D',
            help='dropout probability for encoder input embedding',
        )
        parser.add_argument(
            '--encoder-dropout-out',
            type=float,
            metavar='D',
            help='dropout probability for encoder output',
        )
        parser.add_argument(
            '--decoder-dropout-in',
            type=float,
            metavar='D',
            help='dropout probability for decoder input embedding',
        )
        parser.add_argument(
            '--decoder-dropout-out',
            type=float,
            metavar='D',
            help='dropout probability for decoder output',
        )
        parser.add_argument(
            '--sequence-lstm',
            action='store_true',
            help='use nn.LSTM implementation for encoder',
        )

        # Args for vocab reduction
        vocab_reduction.add_args(parser)
        # Args for word dropout
        word_dropout.add_args(parser)

    @classmethod
    def build_model(cls, args, src_dict, dst_dict):
        """Build a new model instance."""
        base_architecture(args)
        if args.sequence_lstm:
            encoder_class = LSTMSequenceEncoder
        else:
            encoder_class = RNNEncoder
        encoder = encoder_class(
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
            add_encoder_output_as_decoder_input=(
                args.add_encoder_output_as_decoder_input
            ),
        )
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
            add_encoder_output_as_decoder_input=(
                args.add_encoder_output_as_decoder_input
            ),
        )
        return cls(encoder, decoder)

    def get_targets(self, sample, net_output):
        targets = sample['target'].view(-1)
        possible_translation_tokens = net_output[-1]
        if possible_translation_tokens is not None:
            targets = torch_find(
                possible_translation_tokens.data,
                targets.data,
                len(self.dst_dict),
            )
        return targets


class LSTMSequenceEncoder(FairseqEncoder):
    """RNN encoder using nn.LSTM for cuDNN support / ONNX exportability."""

    @staticmethod
    def LSTM(input_size, hidden_size, **kwargs):
        m = nn.LSTM(input_size, hidden_size, **kwargs)
        for name, param in m.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)
        return m

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        freeze_embed=False,
        cell_type='lstm',
        hidden_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        residual_level=None,
        bidirectional=False,
        word_dropout_params=None,
        add_encoder_output_as_decoder_input=False,
    ):
        assert cell_type == 'lstm', 'sequence-lstm requires cell_type="lstm"'

        super().__init__(dictionary)
        self.dictionary = dictionary
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.residual_level = residual_level
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.add_encoder_output_as_decoder_input = (
            add_encoder_output_as_decoder_input
        )
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
            freeze_embed=freeze_embed,
        )
        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            is_layer_bidirectional = self.bidirectional and layer == 0
            self.layers.append(
                LSTMSequenceEncoder.LSTM(
                    embed_dim if layer == 0 else hidden_dim,
                    hidden_dim // 2 if is_layer_bidirectional else hidden_dim,
                    num_layers=1,
                    dropout=self.dropout_out,
                    bidirectional=is_layer_bidirectional,
                )
            )

        self.num_layers = len(self.layers)
        self.word_dropout_module = None
        if (
            word_dropout_params and
            word_dropout_params['word_dropout_freq_threshold'] is not None and
            word_dropout_params['word_dropout_freq_threshold'] > 0
        ):
            self.word_dropout_module = (
                word_dropout.WordDropout(dictionary, word_dropout_params)
            )

    def forward(self, src_tokens, src_lengths):
        if LanguagePairDataset.LEFT_PAD_SOURCE:
            # convert left-padding to right-padding
            src_tokens.data = utils.convert_padding_direction(
                src_tokens.data,
                src_lengths.data,
                self.padding_idx,
                left_to_right=True,
            )
        if self.word_dropout_module is not None:
            src_tokens.data = self.word_dropout_module(src_tokens.data)
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
        packed_input = pack_padded_sequence(
            x,
            src_lengths,
        )

        final_hiddens, final_cells = [], []
        for i, rnn_layer in enumerate(self.layers):
            if self.bidirectional and i == 0:
                h0 = x.data.new(2, bsz, self.hidden_dim // 2).zero_()
                c0 = x.data.new(2, bsz, self.hidden_dim // 2).zero_()
            else:
                h0 = x.data.new(1, bsz, self.hidden_dim).zero_()
                c0 = x.data.new(1, bsz, self.hidden_dim).zero_()

            # apply LSTM along entire sequence
            current_output, (h_last, c_last) = rnn_layer(
                packed_input,
                (h0, c0),
            )

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
        final_hiddens = torch.cat(
            final_hiddens,
            dim=0,
        ).view(self.num_layers, *final_hiddens[0].size())
        final_cells = torch.cat(
            final_cells,
            dim=0,
        ).view(self.num_layers, *final_cells[0].size())

        #  [max_seqlen, batch_size, hidden_dim]
        padding_value = -np.inf if self.add_encoder_output_as_decoder_input else 0
        unpacked_output, _ = pad_packed_sequence(
            packed_input,
            padding_value=padding_value,
        )

        return (
            unpacked_output,
            final_hiddens,
            final_cells,
            src_lengths,
            src_tokens,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class VariableLengthRecurrent(nn.Module):
    '''
    This class acts as a generator of autograd for varying seq lengths with
    different padding behaviors, such as right padding, and order of seq lengths,
    such as descending order.

    The logic is mostly inspired from torch/nn/_functions/rnn.py, so it may be
    merged in the future.
    '''
    def __init__(self, rnn_cell, reverse=False):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.reverse = reverse

    def forward(self, x, hidden, batch_size_per_step):
        self.batch_size_per_step = batch_size_per_step
        self.starting_batch_size = batch_size_per_step[-1] if self.reverse \
            else batch_size_per_step[0]

        output = []
        input_offset = x.size(0) if self.reverse else 0

        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden, )
        initial_hidden = hidden

        if self.reverse:
            hidden = tuple(h[:self.batch_size_per_step[-1]] for h in hidden)

        last_batch_size = self.starting_batch_size

        # Iterate over time steps with varying batch_size
        for i in range(len(self.batch_size_per_step)):
            if self.reverse:
                step_batch_size = self.batch_size_per_step[-1 - i]
                step_input = x[(input_offset - step_batch_size): input_offset]
                input_offset -= step_batch_size
            else:
                step_batch_size = self.batch_size_per_step[i]
                step_input = x[input_offset: (input_offset + step_batch_size)]
                input_offset += step_batch_size

            new_pads = last_batch_size - step_batch_size
            if new_pads > 0:
                # First slice out the pieces for pads
                hiddens.insert(0, tuple(h[-new_pads:] for h in hidden))
                # Only pass the non-pad part of hidden states
                hidden = tuple(h[:-new_pads] for h in hidden)
            if new_pads < 0:
                hidden = tuple(
                    torch.cat((h, ih[last_batch_size:step_batch_size]), 0)
                    for h, ih in zip(hidden, initial_hidden)
                )

            last_batch_size = step_batch_size
            if flat_hidden:
                hidden = (self.rnn_cell(step_input, hidden[0]), )
            else:
                hidden = self.rnn_cell(step_input, hidden)
            output.append(hidden[0])

        if not self.reverse:
            hiddens.insert(0, hidden)
            hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))

        assert output[0].size(0) == self.starting_batch_size

        if flat_hidden:
            hidden = hidden[0]
        if self.reverse:
            output.reverse()

        output = torch.cat(output, 0)
        return hidden, output


class RNNLayer(nn.Module):
    '''
    A wrapper of rnn cells, with their corresponding forward function.
    If bidirectional, halve the hidden_size for each cell.
    '''
    def __init__(
        self,
        input_size,
        hidden_size,
        cell_type='lstm',
        is_bidirectional=False,
    ):
        super().__init__()
        self.is_bidirectional = is_bidirectional
        num_directions = 2 if is_bidirectional else 1

        if cell_type == 'lstm':
            cell_class = rnn_cell.LSTMCell
        elif cell_type == 'milstm':
            cell_class = rnn_cell.MILSTMCell
        elif cell_type == 'layer_norm_lstm':
            cell_class = rnn_cell.LayerNormLSTMCell
        else:
            raise Exception('{} not implemented'.format(cell_type))

        self.fwd_cell = cell_class(input_size, hidden_size // num_directions)
        if is_bidirectional:
            self.bwd_cell = cell_class(input_size, hidden_size // num_directions)

        self.fwd_func = VariableLengthRecurrent(
            rnn_cell=self.fwd_cell,
            reverse=False,
        )
        if is_bidirectional:
            self.bwd_func = VariableLengthRecurrent(
                rnn_cell=self.bwd_cell,
                reverse=True,
            )

    def forward(self, x, hidden, batch_size_per_step):
        fwd_hidden, fwd_output = self.fwd_func.forward(
            x,
            hidden,
            batch_size_per_step,
        )
        if self.is_bidirectional:
            bwd_hidden, bwd_output = self.bwd_func.forward(
                x,
                hidden,
                batch_size_per_step,
            )
            # concat hidden and outputs
            combined_hidden = [fwd_hidden, bwd_hidden]
            bi_hiddens, bi_cells = zip(*combined_hidden)
            next_hidden = (
                torch.cat(bi_hiddens, bi_hiddens[0].dim() - 1),
                torch.cat(bi_cells, bi_cells[0].dim() - 1),
            )
            output = torch.cat([fwd_output, bwd_output], x.dim() - 1)
        else:
            next_hidden = fwd_hidden
            output = fwd_output

        return next_hidden, output


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
        cell_type='lstm',
        dropout_in=0.1,
        dropout_out=0.1,
        residual_level=None,
        bidirectional=False,
        add_encoder_output_as_decoder_input=False,
    ):
        super().__init__(dictionary)
        self.dictionary = dictionary
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.residual_level = residual_level
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.add_encoder_output_as_decoder_input = (
            add_encoder_output_as_decoder_input
        )
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
            freeze_embed=freeze_embed,
        )
        self.cell_type = cell_type
        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            self.layers.append(
                RNNLayer(
                    embed_dim if layer == 0 else hidden_dim,
                    hidden_dim,
                    self.cell_type,
                    True if bidirectional and layer == 0 else False
                )
            )

        self.num_layers = len(self.layers)
        self.word_dropout_module = None
        if (
            word_dropout_params and
            word_dropout_params['word_dropout_freq_threshold'] is not None and
            word_dropout_params['word_dropout_freq_threshold'] > 0
        ):
            self.word_dropout_module = (
                word_dropout.WordDropout(dictionary, word_dropout_params)
            )

    def forward(self, src_tokens, src_lengths):
        if LanguagePairDataset.LEFT_PAD_SOURCE:
            # convert left-padding to right-padding
            src_tokens.data = utils.convert_padding_direction(
                src_tokens.data,
                src_lengths.data,
                self.padding_idx,
                left_to_right=True,
            )
        if self.word_dropout_module is not None:
            src_tokens.data = self.word_dropout_module(src_tokens.data)
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Generate packed seq to deal with varying source seq length
        packed_input, batch_sizes = pack_padded_sequence(
            x,
            src_lengths,
        )
        final_hiddens, final_cells = [], []
        next_hiddens = []
        for i, rnn_layer in enumerate(self.layers):
            current_hidden_size = self.hidden_dim // 2 if \
                rnn_layer.is_bidirectional else self.hidden_dim
            if self.cell_type in ['lstm', 'milstm', 'layer_norm_lstm']:
                prev_hidden = (
                    x.data.new(bsz, current_hidden_size).zero_(),
                    x.data.new(bsz, current_hidden_size).zero_(),
                )
            else:
                raise Exception('{} not implemented'.format(self.cell_type))

            hidden, current_output = rnn_layer.forward(
                packed_input,
                prev_hidden,
                batch_sizes,
            )
            next_hiddens.append(hidden)
            prev_hidden = next_hiddens[-1]

            if self.dropout_out != 0:
                current_output = F.dropout(
                    current_output,
                    p=self.dropout_out,
                    training=self.training,
                )

            if self.residual_level is not None and i >= self.residual_level:
                packed_input = packed_input.clone() + current_output
            else:
                packed_input = current_output

        final_hiddens, final_cells = zip(*next_hiddens)
        # Reshape to [num_layer, batch_size, hidden_dim]
        final_hiddens = torch.cat(
            final_hiddens,
            dim=0,
        ).view(self.num_layers, *final_hiddens[0].size())
        final_cells = torch.cat(
            final_cells,
            dim=0,
        ).view(self.num_layers, *final_cells[0].size())

        #  [max_seqlen, batch_size, hidden_dim]
        padding_value = -np.inf if self.add_encoder_output_as_decoder_input else 0
        unpacked_output, _ = pad_packed_sequence(
            PackedSequence(packed_input, batch_sizes),
            padding_value=padding_value,
        )

        return (
            unpacked_output,
            final_hiddens,
            final_cells,
            src_lengths,
            src_tokens,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class AttentionLayer(nn.Module):
    SUPPORTED_ATTENTION_TYPES = ['dot']

    def __init__(
        self,
        decoder_hidden_state_dim,
        encoder_output_dim,
        attention_type,
    ):
        super().__init__()

        self.decoder_hidden_state_dim = decoder_hidden_state_dim
        self.encoder_output_dim = encoder_output_dim

        assert attention_type in AttentionLayer.SUPPORTED_ATTENTION_TYPES
        self.attention_type = attention_type

        if decoder_hidden_state_dim != encoder_output_dim:
            self.input_proj = Linear(
                decoder_hidden_state_dim,
                encoder_output_dim,
                bias=True,
            )

        # can be externally set to avoid this step for single-example inference
        self.src_length_masking = True

    def forward(self, decoder_state, source_hids, src_lengths):
        """
        Input
            decoder_state: bsz x decoder_hidden_state_dim
            source_hids: srclen x bsz x encoder_output_dim
            src_lengths: bsz x 1, actual sequence lengths
        Output
            output: bsz x encoder_output_dim
            attn_scores: max_srclen x bsz
        """
        if self.attention_type == 'dot':
            output, attn_scores = self.dot_attention(
                decoder_state,
                source_hids,
                src_lengths,
            )
        else:
            raise ValueError(
                'Attention type {} is not supported'.format(self.attention_type)
            )
        return output, attn_scores

    def dot_attention(self, decoder_state, source_hids, src_lengths):
        # decoder_state: bsz x encoder_output_dim
        if self.decoder_hidden_state_dim != self.encoder_output_dim:
            decoder_state = self.input_proj(decoder_state)
        # compute attention
        attn_scores = (source_hids * decoder_state.unsqueeze(0)).sum(dim=2)
        attn_scores = F.softmax(attn_scores.t(), dim=-1)  # bsz x srclen

        if self.src_length_masking:
            # Since input of varying lengths, need to make sure the attn_scores
            # for each sentence sum up to one
            max_srclen = source_hids.size()[0]
            assert max_srclen == src_lengths.data.max()
            batch_size = attn_scores.size()[0]
            src_indices = torch.arange(
                0,
                max_srclen,
            ).unsqueeze(0).type_as(src_lengths.data)
            src_indices = src_indices.expand(batch_size, max_srclen)

            # expand from shape (batch_size,) to (batch_size, max_srclen)
            src_lengths = src_lengths.unsqueeze(dim=1).expand(
                batch_size,
                max_srclen,
            )
            src_mask = (src_indices < src_lengths.data).double().type_as(
                source_hids.data,
            ).detach()
            masked_attn_scores = attn_scores * src_mask
            score_denom = torch.sum(
                masked_attn_scores,
                dim=1,
            ).unsqueeze(dim=1).expand(batch_size, max_srclen)
            normalized_masked_attn_scores = torch.div(
                masked_attn_scores,
                score_denom,
            ).t()
        else:
            normalized_masked_attn_scores = attn_scores.t()

        # sum weighted sources
        attn_weighted_context = (
            source_hids * normalized_masked_attn_scores.unsqueeze(2)
        ).sum(dim=0)

        return attn_weighted_context, normalized_masked_attn_scores


class RNNDecoder(FairseqIncrementalDecoder):
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
        cell_type='lstm',
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        attention_type='dot',
        residual_level=None,
        averaging_encoder=False,
        add_encoder_output_as_decoder_input=False,
    ):
        super().__init__(dst_dict)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.embed_dim = embed_dim
        self.add_encoder_output_as_decoder_input = add_encoder_output_as_decoder_input
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

        if cell_type == 'lstm':
            cell_class = rnn_cell.LSTMCell
        elif cell_type == 'milstm':
            cell_class = rnn_cell.MILSTMCell
        elif cell_type == 'layer_norm_lstm':
            cell_class = rnn_cell.LayerNormLSTMCell

        layers = []
        for layer in range(num_layers):
            if layer == 0:
                cell_input_dim = encoder_hidden_dim + embed_dim
                if self.add_encoder_output_as_decoder_input:
                    cell_input_dim += encoder_hidden_dim
            else:
                cell_input_dim = hidden_dim
            layers.append(
                cell_class(input_dim=cell_input_dim, hidden_dim=hidden_dim))
        self.layers = nn.ModuleList(layers)

        self.attention = AttentionLayer(
            decoder_hidden_state_dim=hidden_dim,
            encoder_output_dim=encoder_hidden_dim,
            attention_type=attention_type,
        )
        self.combined_output_and_context_dim = encoder_hidden_dim + hidden_dim

        self.initial_attn_context = nn.Parameter(
            torch.Tensor(encoder_hidden_dim).zero_(),
        )

        if self.combined_output_and_context_dim != out_embed_dim:
            self.additional_fc = Linear(
                self.combined_output_and_context_dim,
                out_embed_dim,
            )

        self.vocab_reduction_module = None
        if vocab_reduction_params:
            self.vocab_reduction_module = vocab_reduction.VocabReduction(
                src_dict,
                dst_dict,
                vocab_reduction_params,
            )

        self.output_projection_w = nn.Parameter(
            torch.FloatTensor(
                num_embeddings,
                out_embed_dim,
            ).uniform_(-0.1, 0.1),
        )
        self.output_projection_b = nn.Parameter(
            torch.FloatTensor(num_embeddings,).zero_(),
        )

    def forward(
        self,
        input_tokens,
        encoder_out,
        incremental_state=None,
        possible_translation_tokens=None,
    ):
        if incremental_state is not None:
            input_tokens = input_tokens[:, -1:]
        bsz, seqlen = input_tokens.size()

        # get outputs from encoder
        (
            encoder_outs,
            final_hidden,
            final_cell,
            src_lengths,
            src_tokens,
        ) = encoder_out

        # embed tokens
        x = self.embed_tokens(input_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if self.add_encoder_output_as_decoder_input:
            # [1, batch_size, hidden_size]
            encoder_outs_maxpool, _ = torch.max(encoder_outs, dim=0, keepdim=True)
            encoder_outs_maxpool = encoder_outs_maxpool.repeat(seqlen, 1, 1)
            # T x B x (C + encoder_hidden_dim)
            x = torch.cat((x, encoder_outs_maxpool), dim=2)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(
            self,
            incremental_state,
            'cached_state',
        )
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            # first time step, initialize previous states
            prev_hiddens, prev_cells = self._init_prev_states(
                encoder_out,
            )
            input_feed = self.initial_attn_context.expand(
                bsz,
                self.encoder_hidden_dim,
            )

        attn_scores_per_step = []
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            step_input = torch.cat((x[j, :, :], input_feed), dim=1)
            previous_layer_input = step_input
            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(step_input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                layer_output = F.dropout(
                    hidden,
                    p=self.dropout_out,
                    training=self.training,
                )

                if self.residual_level is not None and i >= self.residual_level:
                    # TODO(T25321141) add an assert related to sizes here
                    step_input = layer_output + previous_layer_input
                else:
                    step_input = layer_output
                previous_layer_input = step_input

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out, step_attn_scores = self.attention(
                hidden,
                encoder_outs,
                src_lengths,
            )
            attn_scores_per_step.append(step_attn_scores.unsqueeze(1))
            combined_output_and_context = torch.cat((hidden, out), dim=1)

            # input feeding
            input_feed = out

            # save final output (1 x B x C)
            outs.append(combined_output_and_context.unsqueeze(0))

        attn_scores = torch.cat(attn_scores_per_step, dim=1)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self,
            incremental_state,
            'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        x = torch.cat(outs, dim=0)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        # bottleneck layer
        if hasattr(self, 'additional_fc'):
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)

        output_projection_w = self.output_projection_w
        output_projection_b = self.output_projection_b
        decoder_input_tokens = input_tokens if self.training else None

        if self.vocab_reduction_module and possible_translation_tokens is None:
            possible_translation_tokens = self.vocab_reduction_module(
                src_tokens,
                decoder_input_tokens=decoder_input_tokens,
            )

        if possible_translation_tokens is not None:
            output_projection_w = output_projection_w.index_select(
                dim=0,
                index=possible_translation_tokens,
            )
            output_projection_b = output_projection_b.index_select(
                dim=0,
                index=possible_translation_tokens,
            )

        # avoiding transpose of projection weights during ONNX tracing
        batch_time_hidden = torch.onnx.operators.shape_as_tensor(x)
        x_flat_shape = torch.cat(
            (
                torch.LongTensor([-1]),
                batch_time_hidden[2].view(1),

            ),
        )
        x_flat = torch.onnx.operators.reshape_from_tensor_shape(
            x,
            x_flat_shape,
        )

        projection_flat = torch.matmul(output_projection_w, x_flat.t()).t()
        logits_shape = torch.cat(
            (
                batch_time_hidden[:2],
                torch.LongTensor([-1]),
            ),
        )
        logits = torch.onnx.operators.reshape_from_tensor_shape(
            projection_flat,
            logits_shape,
        ) + output_projection_b

        return logits, attn_scores, possible_translation_tokens

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        cached_state = utils.get_incremental_state(
            self,
            incremental_state,
            'cached_state',
        )
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

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
        return prev_hiddens, prev_cells


def Embedding(num_embeddings, embedding_dim, padding_idx, freeze_embed):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    if freeze_embed:
        m.weight.requires_grad = False
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture('rnn', 'rnn')
def base_architecture(args):
    # default architecture
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 512)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 512)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.attention_type = getattr(args, 'attention_type', 'dot')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.averaging_encoder = getattr(args, 'averaging_encoder', False)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.cell_type = getattr(args, 'cell_type', 'lstm')
    vocab_reduction.set_arg_defaults(args)
    word_dropout.set_arg_defaults(args)
    args.sequence_lstm = getattr(args, 'sequence_lstm', False)
    args.add_encoder_output_as_decoder_input = getattr(
        args,
        'add_encoder_output_as_decoder_input',
        False,
    )


@register_model_architecture('rnn', 'rnn_big_test')
def rnn_big_test(args):
    base_architecture(args)
    args.encoder_embed_dim = 1024
    args.encoder_layers = 6
    args.encoder_hidden_dim = 1024
    args.decoder_embed_dim = 1024
    args.decoder_layers = 6
    args.decoder_hidden_dim = 1024
    args.decoder_out_embed_dim = 1024
