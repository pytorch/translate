#!/usr/bin/env python3

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from pytorch_translate import rnn_cell  # noqa


class AttentionLayer(nn.Module):
    SUPPORTED_ATTENTION_TYPES = ["dot", "max", "mean"]

    def __init__(
        self,
        decoder_hidden_state_dim,
        encoder_output_dim,
        attention_type,
        force_projection=False,
    ):
        super().__init__()

        self.decoder_hidden_state_dim = decoder_hidden_state_dim
        self.encoder_output_dim = encoder_output_dim

        assert attention_type in AttentionLayer.SUPPORTED_ATTENTION_TYPES
        self.attention_type = attention_type
        self.context_dim = encoder_output_dim

        self.input_proj = None
        if force_projection or decoder_hidden_state_dim != encoder_output_dim:
            self.input_proj = Linear(
                decoder_hidden_state_dim, encoder_output_dim, bias=True
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
        if self.attention_type == "dot":
            output, attn_scores = self.dot_attention(
                decoder_state, source_hids, src_lengths
            )
        elif self.attention_type == 'mean' or self.attention_type == 'max':
            output, attn_scores = self.pooling(
                self.attention_type,
                source_hids,
                src_lengths,
            )
        else:
            raise ValueError(f"Attention type {self.attention_type} is not supported")
        return output, attn_scores

    def pooling(self, pool_type, source_hids, src_lengths):
        assert self.decoder_hidden_state_dim == self.encoder_output_dim
        max_srclen = source_hids.size()[0]
        assert max_srclen == src_lengths.data.max()
        batch_size = source_hids.size()[1]
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
        src_mask = Variable(
            (src_indices < src_lengths.data).double().type_as(
                source_hids.data,
            ),
            requires_grad=False,
        ).t().unsqueeze(2)
        denom = src_lengths.t().unsqueeze(2).type_as(source_hids)
        if pool_type == "mean":
            masked_hiddens = source_hids * src_mask
            context = (masked_hiddens / denom).sum(dim=0)
        elif pool_type == "max":
            masked_hiddens = source_hids - 10e6 * (1 - src_mask)
            context = masked_hiddens.max(dim=0)
        else:
            raise ValueError(
                'Pooling type {} is not supported'.format(pool_type)
            )
        attn_scores = Variable(
            torch.ones(src_mask.shape[1], src_mask.shape[0]).type_as(
                source_hids.data,
            ),
            requires_grad=False,
        ).t()
        return context, attn_scores

    def dot_attention(self, decoder_state, source_hids, src_lengths):
        # decoder_state: bsz x encoder_output_dim
        if self.input_proj is not None:
            decoder_state = self.input_proj(decoder_state)
        # compute attention
        attn_scores = (source_hids * decoder_state.unsqueeze(0)).sum(dim=2)

        if self.src_length_masking:
            max_srclen = source_hids.size()[0]
            assert max_srclen == src_lengths.data.max()
            batch_size = source_hids.size()[1]
            src_indices = torch.arange(0, max_srclen).unsqueeze(1).type_as(
                src_lengths.data
            )
            src_indices = src_indices.expand(max_srclen, batch_size)

            # expand from shape (batch_size,) to (max_srclen, batch_size)
            src_lengths = src_lengths.unsqueeze(dim=0).expand(max_srclen, batch_size)
            src_mask = (src_indices < src_lengths.data).double().type_as(
                source_hids.data
            ).detach()
            masked_attn_scores = torch.where(
                src_mask > 0,
                attn_scores,
                torch.Tensor([float("-Inf")]).type_as(source_hids.data),
            )
            # Since input of varying lengths, need to make sure the attn_scores
            # for each sentence sum up to one
            attn_scores = F.softmax(masked_attn_scores.t(), dim=-1)  # bsz x srclen
            score_denom = torch.sum(attn_scores, dim=1).unsqueeze(dim=1).expand(
                batch_size, max_srclen
            )
            normalized_masked_attn_scores = torch.div(attn_scores, score_denom).t()
        else:
            normalized_masked_attn_scores = F.softmax(attn_scores.t(), dim=-1).t()

        # sum weighted sources
        attn_weighted_context = (
            source_hids * normalized_masked_attn_scores.unsqueeze(2)
        ).sum(
            dim=0
        )

        return attn_weighted_context, normalized_masked_attn_scores


class VariableLengthRecurrent(nn.Module):
    """
    This class acts as a generator of autograd for varying seq lengths with
    different padding behaviors, such as right padding, and order of seq lengths,
    such as descending order.

    The logic is mostly inspired from torch/nn/_functions/rnn.py, so it may be
    merged in the future.
    """

    def __init__(self, rnn_cell, reverse=False):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.reverse = reverse

    def forward(self, x, hidden, batch_size_per_step):
        self.batch_size_per_step = batch_size_per_step
        self.starting_batch_size = batch_size_per_step[
            -1
        ] if self.reverse else batch_size_per_step[
            0
        ]

        output = []
        input_offset = x.size(0) if self.reverse else 0

        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        initial_hidden = hidden

        if self.reverse:
            hidden = tuple(h[: self.batch_size_per_step[-1]] for h in hidden)

        last_batch_size = self.starting_batch_size

        # Iterate over time steps with varying batch_size
        for i in range(len(self.batch_size_per_step)):
            if self.reverse:
                step_batch_size = self.batch_size_per_step[-1 - i]
                step_input = x[(input_offset - step_batch_size) : input_offset]
                input_offset -= step_batch_size
            else:
                step_batch_size = self.batch_size_per_step[i]
                step_input = x[input_offset : (input_offset + step_batch_size)]
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
                hidden = (self.rnn_cell(step_input, hidden[0]),)
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
    """
    A wrapper of rnn cells, with their corresponding forward function.
    If bidirectional, halve the hidden_size for each cell.
    """

    def __init__(
        self, input_size, hidden_size, cell_type="lstm", is_bidirectional=False
    ):
        super().__init__()
        self.is_bidirectional = is_bidirectional
        num_directions = 2 if is_bidirectional else 1

        if cell_type == "lstm":
            cell_class = rnn_cell.LSTMCell
        elif cell_type == "milstm":
            cell_class = rnn_cell.MILSTMCell
        elif cell_type == "layer_norm_lstm":
            cell_class = rnn_cell.LayerNormLSTMCell
        else:
            raise Exception(f"{cell_type} not implemented")

        self.fwd_cell = cell_class(input_size, hidden_size // num_directions)
        if is_bidirectional:
            self.bwd_cell = cell_class(input_size, hidden_size // num_directions)

        self.fwd_func = VariableLengthRecurrent(rnn_cell=self.fwd_cell, reverse=False)
        if is_bidirectional:
            self.bwd_func = VariableLengthRecurrent(
                rnn_cell=self.bwd_cell, reverse=True
            )

    def forward(self, x, hidden, batch_size_per_step):
        fwd_hidden, fwd_output = self.fwd_func.forward(x, hidden, batch_size_per_step)
        if self.is_bidirectional:
            bwd_hidden, bwd_output = self.bwd_func.forward(
                x, hidden, batch_size_per_step
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


def Embedding(num_embeddings, embedding_dim, padding_idx, freeze_embed):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    if freeze_embed:
        m.weight.requires_grad = False
    return m


def Linear(in_features, out_features, bias=True):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


def NonlinearLayer(in_features, out_features, bias=True, activation_fn=nn.ReLU):
    """Weight-normalized non-linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return nn.Sequential(m, activation_fn())
