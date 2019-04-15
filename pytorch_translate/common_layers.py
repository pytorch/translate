#!/usr/bin/env python3

import abc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder, transformer as fairseq_transformer
from pytorch_translate import rnn_cell  # noqa
from pytorch_translate import utils as pytorch_translate_utils, vocab_reduction
from pytorch_translate.research.lexical_choice import lexical_translation


class ContextEmbedding(nn.Module):
    """
    This class implements context-dependent word embeddings as described in
    https://arxiv.org/pdf/1607.00578.pdf
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.nonlinear = NonlinearLayer(
            embed_dim, embed_dim, bias=True, activation_fn=nn.ReLU
        )
        self.linear = Linear(embed_dim, embed_dim, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, src):
        c = torch.mean(self.nonlinear(src), 1, True)
        return src * self.sigmoid(self.linear(c))


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


class Embedding(nn.Embedding):
    """
    A wrapper around the embedding layer, which can be randomly initialized or
    loaded from a .npy file. Also supports normalization of embeddings to have
    zero mean and unit variance (weighted by token frequency) - this is useful
    for example when creating adversarial perturbations of the embeddings that
    should have norms relative to the embeddings' norms.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx,
        freeze_embed=False,
        normalize_embed=False,
        normalize_decay_rate=0.99,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        nn.init.uniform_(self.weight, -0.1, 0.1)
        nn.init.constant_(self.weight[padding_idx], 0.0)
        if freeze_embed:
            self.weight.requires_grad = False

        assert 0.0 < normalize_decay_rate < 1.0
        self.normalize = normalize_embed
        self.normalize_decay_rate = normalize_decay_rate
        self.mean = None
        self.var = None
        self.init_normalization_if_needed()

    def forward(self, x):
        x = super().forward(x)

        if self.normalize:
            if self.training:
                self._update_normalize_params(x)
            x = (x - self.mean) / torch.sqrt(self.var + 1e-6)

        return x

    def init_normalization_if_needed(self):
        if not self.normalize:
            return

        self.mean = nn.Parameter(self.weight.mean(dim=0), requires_grad=False)
        self.var = nn.Parameter(self.weight.var(dim=0), requires_grad=False)

    def _update_normalize_params(self, x):
        """
        Updates the observed mean and variance of the token embeddings. Note
        that these will be weighted by the empirical frequency of each token
        (i.e. common tokens will be more heavily weighted in the params).
        """
        # Flatten x to be a tensor of embeddings.
        assert x.size()[-1:] == self.mean.size()
        x_flattened = x.view(-1, x.size(-1))

        # Update mean.
        x_mean = x_flattened.mean(dim=0)
        self.mean.data = (
            self.normalize_decay_rate * self.mean.data
            + (1.0 - self.normalize_decay_rate) * x_mean
        )

        # Update variance.
        x_var = ((x_flattened - self.mean) ** 2).mean(dim=0)
        self.var.data = (
            self.normalize_decay_rate * self.var.data
            + (1.0 - self.normalize_decay_rate) * x_var
        )


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
        out_embed_norm=None,
        att_weighted_src_embeds=False,
        src_embed_dim=512,
        att_weighted_activation_type="tanh",
        predictor=None,
        fp16: bool = False,
    ):
        super().__init__(dst_dict)
        self.project_output = project_output
        if project_output:
            self.num_embeddings = len(dst_dict)
            self.out_embed_dim = out_embed_dim
            self.out_embed_norm = out_embed_norm
            self.att_weighted_src_embeds = att_weighted_src_embeds
            self.src_embed_dim = src_embed_dim
            self.vocab_reduction_module = None
            if vocab_reduction_params or predictor is not None:
                self.vocab_reduction_module = vocab_reduction.VocabReduction(
                    src_dict=src_dict,
                    dst_dict=dst_dict,
                    vocab_reduction_params=vocab_reduction_params,
                    predictor=predictor,
                    fp16=fp16,
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
            if att_weighted_activation_type == "tanh":
                activation_fn = nn.Tanh
                self.att_weighted_activation_fn = torch.tanh
            elif att_weighted_activation_type == "relu":
                activation_fn = nn.ReLU
                self.att_weighted_activation_fn = torch.relu
            else:
                raise Exception(
                    "att_weighted_activation_type '%s' not implemented"
                    % att_weighted_activation_type
                )
            if att_weighted_src_embeds:
                print(att_weighted_activation_type)
                self.lexical_layer = NonlinearLayer(
                    self.src_embed_dim,
                    self.out_embed_dim,
                    bias=False,
                    activation_fn=activation_fn,
                )
                self.output_projection_w_lex = nn.Parameter(
                    torch.FloatTensor(self.num_embeddings, self.out_embed_dim).uniform_(
                        -0.1, 0.1
                    )
                )
                self.output_projection_b_lex = nn.Parameter(
                    torch.FloatTensor(self.num_embeddings).zero_()
                )

    def forward(
        self,
        input_tokens,
        encoder_out,
        incremental_state=None,
        possible_translation_tokens=None,
        reduced_output_weights=None,
    ):
        (_, _, _, src_lengths, src_tokens, src_embeddings) = encoder_out
        x, attn_scores = self.forward_unprojected(
            input_tokens, encoder_out, incremental_state
        )
        if not self.project_output:
            return x, attn_scores, None
        decoder_input_tokens = input_tokens.contiguous()

        if reduced_output_weights is not None:
            (output_projection_w, output_projection_b) = reduced_output_weights
        else:
            output_projection_w = self.output_projection_w
            output_projection_b = self.output_projection_b

            if self.vocab_reduction_module and possible_translation_tokens is None:
                possible_translation_tokens = self.vocab_reduction_module(
                    src_tokens,
                    encoder_output=encoder_out,
                    decoder_input_tokens=decoder_input_tokens,
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

        if self.out_embed_norm is not None:
            # fix the norm of both output word embeddings and context vector
            output_projection_w = self.out_embed_norm * F.normalize(
                output_projection_w, p=2, dim=1
            )
            x_flat = self.out_embed_norm * F.normalize(x_flat, p=2, dim=1)

        projection_flat = torch.matmul(output_projection_w, x_flat.t()).t()
        logits_shape = torch.cat((batch_time_hidden[:2], torch.LongTensor([-1])))
        logits = (
            torch.onnx.operators.reshape_from_tensor_shape(
                projection_flat, logits_shape
            )
            + output_projection_b
        )
        if self.att_weighted_src_embeds:
            # use the attention weights to form a weighted average of embeddings
            lex = lexical_translation.attention_weighted_src_embedding(
                src_embeddings, attn_scores, self.att_weighted_activation_fn
            )
            # avoiding transpose of projection weights during ONNX tracing
            batch_time_hidden_lex = torch.onnx.operators.shape_as_tensor(lex)
            lex_flat_shape = torch.cat(
                (torch.LongTensor([-1]), batch_time_hidden_lex[2].view(1))
            )
            lex_flat = torch.onnx.operators.reshape_from_tensor_shape(
                lex, lex_flat_shape
            )
            lex_logits_shape = torch.cat(
                (batch_time_hidden_lex[:2], torch.LongTensor([-1]))
            )
            # add one-hidden-layer FFNN
            lex_h = self.lexical_layer(lex_flat)
            # lexical logits of output of previous FFNN
            lex_logits = lexical_translation.lex_logits(
                lex_h,
                self.output_projection_w_lex,
                self.output_projection_b_lex,
                lex_logits_shape,
            )
            # combine lexical logits with the original decoder logits
            logits.add_(lex_logits)

        return logits, attn_scores, possible_translation_tokens

    def _precompute_reduced_weights(self, possible_translation_tokens):
        output_projection_w = self.output_projection_w.index_select(
            dim=0, index=possible_translation_tokens
        )
        output_projection_b = self.output_projection_b.index_select(
            dim=0, index=possible_translation_tokens
        )
        return (output_projection_w, output_projection_b)

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


class VariableTracker(object):
    """This class implements several methods to keep track of intermediate
    variables.

    This is useful for eg. visualizing or retrieving gradients wrt. inputs
    later on"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tracker = {}

    def track(self, variable, name, retain_grad=False):
        """Adds variable to the tracker

        Specify `retain_grad=True` to retrieve the gradient later."""
        if retain_grad:
            variable.retain_grad()
        self.tracker[name] = variable

    def __getitem__(self, name):
        return self.tracker[name]


def build_embedding(dictionary, embed_dim, path=None):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)
    # if provided, load from preloaded dictionaries
    if path:
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, emb)
    return emb


class TransformerEncoderGivenEmbeddings(nn.Module):
    def __init__(self, args, proj_to_decoder):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                fairseq_transformer.TransformerEncoderLayer(args)
                for i in range(args.encoder_layers)
            ]
        )

        self.output_fc = None
        if args.encoder_embed_dim != args.decoder_embed_dim and proj_to_decoder:
            self.output_fc = fairseq_transformer.Linear(
                args.encoder_embed_dim, args.decoder_embed_dim
            )

    def forward(self, x, positions, encoder_padding_mask):
        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.output_fc is not None:
            x = self.output_fc(x)

        return x

    def upgrade_state_dict_named(self, state_dict, name):
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, f"{name}.layers.{i}")


def TransformerTokenEmbedding(
    num_embeddings, embedding_dim, padding_idx, freeze_embed=False
):
    """
    Different weight initialization from common_layers.Embedding
    """
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    if freeze_embed:
        m.weight.requires_grad = False
    return m


class TransformerEmbedding(nn.Module):
    def __init__(self, args, embed_tokens):
        super().__init__()
        self.dropout = args.dropout
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = fairseq_transformer.PositionalEmbedding(
            1024,
            embed_dim,
            self.padding_idx,
            learned=args.encoder_learned_pos,
        )

    def forward(self, src_tokens, src_lengths):
        # Embed tokens
        x = self.embed_tokens(src_tokens)
        src_tokens_tensor = pytorch_translate_utils.get_source_tokens_tensor(src_tokens)
        # Add position embeddings and dropout
        x = self.embed_scale * x
        positions = self.embed_positions(src_tokens_tensor)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask (B x T)
        encoder_padding_mask = src_tokens_tensor.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        return x, encoder_padding_mask, positions
