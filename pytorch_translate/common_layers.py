#!/usr/bin/env python3

import abc

import numpy as np
import torch
import torch.nn as nn
from fairseq.models import FairseqIncrementalDecoder
from fairseq import utils
from pytorch_translate import rnn_cell  # noqa
from pytorch_translate import vocab_reduction


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
        self.starting_batch_size = (
            batch_size_per_step[-1] if self.reverse else batch_size_per_step[0]
        )

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


def Embedding(
    num_embeddings, embedding_dim, padding_idx, freeze_embed
):
    """
    A wrapper around the embedding layer, which can be randomly
    initialized or loaded from a .npy file.
    """

    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0.)
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


class DecoderWithOutputProjection(FairseqIncrementalDecoder):
    """Common super class for decoder networks with output projection layers.

    This class couples common functionality for `FairseqDecoder`s with large
    output projection layers such as ONNX compatibility and vocabulary reduction.
    """

    def __init__(
        self,
        src_dict,
        dst_dict,
        vocab_reduction_params=None,
        out_embed_dim=512,
        project_output=True,
        pretrained_embed=None,
    ):
        super().__init__(dst_dict)
        self.project_output = project_output
        if project_output:
            self.num_embeddings = len(dst_dict)
            self.out_embed_dim = out_embed_dim
            self.vocab_reduction_module = None
            if vocab_reduction_params:
                self.vocab_reduction_module = vocab_reduction.VocabReduction(
                    src_dict, dst_dict, vocab_reduction_params
                )

            projection_weights = torch.FloatTensor(
                self.num_embeddings, self.out_embed_dim
            ).uniform_(-0.1, 0.1)
            if isinstance(pretrained_embed, nn.Embedding):
                projection_weights.data = pretrained_embed.weights.data
            elif pretrained_embed is not None:
                embed_dict = utils.parse_embedding(pretrained_embed)
                # equivalent to utils.load_embedding but for nn.Parameter
                for idx in range(len(dst_dict)):
                    token = dst_dict[idx]
                    if token in embed_dict:
                        projection_weights[idx] = embed_dict[token]
            self.output_projection_w = nn.Parameter(projection_weights)
            self.output_projection_b = nn.Parameter(
                torch.FloatTensor(self.num_embeddings).zero_()
            )

    def forward(
        self,
        input_tokens,
        encoder_out,
        incremental_state=None,
        possible_translation_tokens=None,
    ):
        (_, _, _, _, src_tokens) = encoder_out
        x, attn_scores = self.forward_unprojected(
            input_tokens, encoder_out, incremental_state
        )
        if not self.project_output:
            return x, attn_scores, None
        output_projection_w = self.output_projection_w
        output_projection_b = self.output_projection_b
        decoder_input_tokens = input_tokens.contiguous()

        if self.vocab_reduction_module and possible_translation_tokens is None:
            possible_translation_tokens = self.vocab_reduction_module(
                src_tokens, decoder_input_tokens=decoder_input_tokens
            )

        if possible_translation_tokens is not None:
            output_projection_w = output_projection_w.index_select(
                dim=0, index=possible_translation_tokens
            )
            output_projection_b = output_projection_b.index_select(
                dim=0, index=possible_translation_tokens
            )

        # avoiding transpose of projection weights during ONNX tracing
        batch_time_hidden = torch.onnx.operators.shape_as_tensor(x)
        x_flat_shape = torch.cat((torch.LongTensor([-1]), batch_time_hidden[2].view(1)))
        x_flat = torch.onnx.operators.reshape_from_tensor_shape(x, x_flat_shape)

        projection_flat = torch.matmul(output_projection_w, x_flat.t()).t()
        logits_shape = torch.cat((batch_time_hidden[:2], torch.LongTensor([-1])))
        logits = (
            torch.onnx.operators.reshape_from_tensor_shape(
                projection_flat, logits_shape
            )
            + output_projection_b
        )

        return logits, attn_scores, possible_translation_tokens

    @abc.abstractmethod
    def forward_unprojected(
        self,
        input_tokens,
        encoder_out,
        incremental_state=None,
        possible_translation_tokens=None,
    ):
        """Forward pass through the decoder without output projection."""
        raise NotImplementedError()


class OutputProjection(nn.Module):
    """Output projection layer."""

    def __init__(self, out_embed_dim, vocab_size, vocab_reduction_module=None):
        super().__init__()
        self.out_embed_dim = out_embed_dim
        self.vocab_size = vocab_size

        self.output_projection_w = nn.Parameter(
            torch.FloatTensor(self.vocab_size, self.out_embed_dim).uniform_(-0.1, 0.1)
        )
        self.output_projection_b = nn.Parameter(
            torch.FloatTensor(self.vocab_size).zero_()
        )
        self.vocab_reduction_module = vocab_reduction_module

    def forward(
        self, x, src_tokens=None, input_tokens=None, possible_translation_tokens=None
    ):
        output_projection_w = self.output_projection_w
        output_projection_b = self.output_projection_b
        decoder_input_tokens = input_tokens if self.training else None

        if self.vocab_reduction_module and possible_translation_tokens is None:
            possible_translation_tokens = self.vocab_reduction_module(
                src_tokens, decoder_input_tokens=decoder_input_tokens
            )

        if possible_translation_tokens is not None:
            output_projection_w = output_projection_w.index_select(
                dim=0, index=possible_translation_tokens
            )
            output_projection_b = output_projection_b.index_select(
                dim=0, index=possible_translation_tokens
            )

        # avoiding transpose of projection weights during ONNX tracing
        batch_time_hidden = torch.onnx.operators.shape_as_tensor(x)
        x_flat_shape = torch.cat((torch.LongTensor([-1]), batch_time_hidden[2].view(1)))
        x_flat = torch.onnx.operators.reshape_from_tensor_shape(x, x_flat_shape)

        projection_flat = torch.matmul(output_projection_w, x_flat.t()).t()
        logits_shape = torch.cat((batch_time_hidden[:2], torch.LongTensor([-1])))
        logits = (
            torch.onnx.operators.reshape_from_tensor_shape(
                projection_flat, logits_shape
            )
            + output_projection_b
        )
        return logits, possible_translation_tokens
