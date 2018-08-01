#!/usr/bin/env python3

import abc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder
from pytorch_translate import rnn, vocab_reduction
from pytorch_translate.common_layers import Linear, NonlinearLayer, OutputProjection
from pytorch_translate.utils import average_tensors, maybe_cuda
from torch.serialization import default_restore_location


def unfreeze_nth_component(components, unfreeze_idx=-1):
    """Freeze weights in all components except the `unfreeze_idx`-th one."""
    for idx, component in enumerate(components):
        for p in component.parameters():
            p.requires_grad = idx == unfreeze_idx


class MultiEncoder(FairseqEncoder):
    """Concatenates the outputs of multiple encoders."""

    def __init__(self, dictionary, encoders, training_schedule="complete"):
        super().__init__(dictionary)
        self.encoders = nn.ModuleList(encoders)
        self.unfreeze_single = False
        self.unfreeze_idx = -1
        if self.training:
            if training_schedule in ["freeze_all", "freeze_all_encoders"]:
                unfreeze_nth_component(self.encoders)
            elif training_schedule.startswith(
                "unfreeze_enc_"
            ) or training_schedule.startswith("unfreeze_encdec_"):
                _, _, n = training_schedule.split("_")
                unfreeze_nth_component(self.encoders, int(n))
            elif training_schedule in ["unfreeze_single", "unfreeze_single_encoder"]:
                self.unfreeze_single = True
                self.unfreeze_mod = len(encoders)
            elif training_schedule == "separate":
                self.unfreeze_single = True
                self.unfreeze_mod = len(encoders) + 1
            elif training_schedule != "complete":
                raise RuntimeError(f"Unknown training schedule '{training_schedule}'")

    def forward(self, src_tokens, src_lengths):
        if self.unfreeze_single:
            self.unfreeze_idx = (self.unfreeze_idx + 1) % self.unfreeze_mod
            unfreeze_nth_component(self.encoders, self.unfreeze_idx)
        all_encoder_outs = [
            encoder(src_tokens, src_lengths) for encoder in self.encoders
        ]
        combined_encoder_outs = []
        for i in range(3):
            combined_encoder_outs.append(
                torch.cat([e[i] for e in all_encoder_outs], dim=2)
            )
        # src_tokens and src_lengths are taken from the first encoder.
        combined_encoder_outs.extend(all_encoder_outs[0][3:])
        return tuple(combined_encoder_outs)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder all outputs according to new_order."""
        return rnn.reorder_encoder_output(encoder_out, new_order)


class MultiDecoderCombinationStrategy(nn.Module):
    """Strategy for combining decoder networks.

    This is an abstract strategy (GoF) which defines the mapping from multiple
    (unprojected) decoder outputs to the fully expanded logits.
    """

    def __init__(self, out_embed_dims, vocab_size, vocab_reduction_module=None):
        super().__init__()
        self.out_embed_dims = out_embed_dims
        self.vocab_size = vocab_size
        self.vocab_reduction_module = vocab_reduction_module

    @abc.abstractmethod
    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
        select_single=None,
    ):
        """Combine decoder outputs and project.

        Args:
            unprojected_outs (list): List of tensors with the same length as
                self.out_embed_dims containing the unprojected decoder outputs
                from each decoder network.
            src_tokens (Tensor): Tensor with source sentence tokens for vocab
                reduction.
            input_tokens (Tensor): Tensor with target-side decoder input tokens
                for vocab reduction.
            possible_translation_tokens: For vocab reduction.
            select_single (None or int): Only use the n-th decoder output.

        Return:
            A tuple (logits, possible_translation_tokens), where logits is a
            [batch_size, seq_len, vocab_size] tensor with the final combined
            output logits, and possible_translation_tokens the short list from
            vocab reduction.
        """
        raise NotImplementedError()


class UniformStrategy(MultiDecoderCombinationStrategy):
    """Uniform averaging of model predictions."""

    def __init__(
        self,
        out_embed_dims,
        vocab_size,
        vocab_reduction_module=None,
        norm_fn=None,
        to_log=False,
    ):
        super().__init__(out_embed_dims, vocab_size)
        assert vocab_reduction_module is None
        self.output_projections = nn.ModuleList(
            [OutputProjection(dim, vocab_size) for dim in out_embed_dims]
        )
        self.to_log = to_log
        self.norm_fn = norm_fn

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
        select_single=None,
    ):
        assert possible_translation_tokens is None
        if select_single is not None:
            return self.output_projections[select_single](
                unprojected_outs[select_single]
            )
        logits = [p(o)[0] for p, o in zip(self.output_projections, unprojected_outs)]
        avg = average_tensors(logits, norm_fn=self.norm_fn)
        if self.to_log:
            avg.log_()
        return avg, None


class UnprojectedStrategy(MultiDecoderCombinationStrategy):
    """Average decoder outputs, share output projection layer."""

    def __init__(self, out_embed_dims, vocab_size, vocab_reduction_module=None):
        super().__init__(out_embed_dims, vocab_size, vocab_reduction_module)
        out_embed_dim = out_embed_dims[0]
        assert all(d == out_embed_dim for d in out_embed_dims)
        self.output_projection = OutputProjection(
            out_embed_dim, vocab_size, vocab_reduction_module
        )

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
        select_single=None,
    ):
        return self.output_projection(
            average_tensors(unprojected_outs)
            if select_single is None
            else unprojected_outs[select_single],
            src_tokens,
            input_tokens,
            possible_translation_tokens,
        )


class MaxUnprojectedStrategy(MultiDecoderCombinationStrategy):
    """Element-wise max of decoder outputs, share output projection layer."""

    def __init__(self, out_embed_dims, vocab_size, vocab_reduction_module=None):
        out_embed_dim = out_embed_dims[0]
        assert all(d == out_embed_dim for d in out_embed_dims)
        super().__init__(out_embed_dims, vocab_size, vocab_reduction_module)
        self.output_projection = OutputProjection(
            out_embed_dim, vocab_size, vocab_reduction_module
        )

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
        select_single=None,
    ):
        if select_single is None:
            proj_input, _ = torch.max(torch.stack(unprojected_outs), dim=0)
        else:
            proj_input = unprojected_outs[select_single]
        return self.output_projection(
            proj_input, src_tokens, input_tokens, possible_translation_tokens
        )


class MultiplicativeUnprojectedStrategy(MultiDecoderCombinationStrategy):
    """Element-wise product of decoder out, share output projection layer."""

    def __init__(self, out_embed_dims, vocab_size, vocab_reduction_module=None):
        super().__init__(out_embed_dims, vocab_size, vocab_reduction_module)
        out_embed_dim = out_embed_dims[0]
        assert all(d == out_embed_dim for d in out_embed_dims)
        self.output_projection = OutputProjection(
            out_embed_dim, vocab_size, vocab_reduction_module
        )
        self.activation = nn.ReLU()

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
        select_single=None,
    ):
        stacked = (
            torch.stack(unprojected_outs)
            if select_single is None
            else torch.unsqueeze(unprojected_outs[select_single], 0)
        )
        return self.output_projection(
            torch.prod(self.activation(stacked), dim=0),
            src_tokens,
            input_tokens,
            possible_translation_tokens,
        )


class DeepFusionStrategy(MultiDecoderCombinationStrategy):
    """Deep fusion following https://arxiv.org/pdf/1503.03535.pdf.

    The first decoder is assumed to be the language model.
    """

    def __init__(self, out_embed_dims, vocab_size, vocab_reduction_module=None):
        super().__init__(out_embed_dims, vocab_size, vocab_reduction_module)
        self.gating_network = NonlinearLayer(
            out_embed_dims[0], 1, bias=True, activation_fn=nn.Sigmoid
        )
        self.output_projection = OutputProjection(
            sum(out_embed_dims), vocab_size, vocab_reduction_module
        )

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
        select_single=None,
    ):
        assert select_single is None
        g = self.gating_network(unprojected_outs[0])
        unprojected_outs[0] = g * unprojected_outs[0]
        return self.output_projection(
            torch.cat(unprojected_outs, 2),
            src_tokens,
            input_tokens,
            possible_translation_tokens,
        )


class ColdFusionStrategy(MultiDecoderCombinationStrategy):
    """Cold fusion following https://arxiv.org/pdf/1708.06426.pdf.

    The first decoder is assumed to be the language model.
    """

    def __init__(
        self,
        out_embed_dims,
        vocab_size,
        vocab_reduction_module=None,
        hidden_layer_size=256,
    ):
        super().__init__(out_embed_dims, vocab_size, vocab_reduction_module)
        self.hidden_layer = NonlinearLayer(
            vocab_size, hidden_layer_size, bias=False, activation_fn=nn.ReLU
        )
        trans_dim = sum(out_embed_dims[1:])
        self.gating_network = NonlinearLayer(
            hidden_layer_size + trans_dim,
            hidden_layer_size,
            bias=True,
            activation_fn=nn.Sigmoid,
        )

        # output_projections is [LM projection, Joint projection]. This is a
        # trick to load pretrained LM projection.
        self.output_projections = nn.ModuleList(
            [
                OutputProjection(out_embed_dims[0], vocab_size),
                OutputProjection(
                    hidden_layer_size + trans_dim, vocab_size, vocab_reduction_module
                ),
            ]
        )
        self.pre_softmax_activation = nn.ReLU()

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
        select_single=None,
    ):
        assert select_single is None
        l_lm, _ = self.output_projections[0](
            unprojected_outs[0], src_tokens, input_tokens, possible_translation_tokens
        )
        l_lm_max, _ = torch.max(l_lm, dim=2, keepdim=True)
        l_lm = l_lm - l_lm_max
        h_lm = self.hidden_layer(l_lm)
        s = torch.cat(unprojected_outs[1:], 2)
        g = self.gating_network(torch.cat([s, h_lm], 2))
        s_cf = torch.cat([s, g * h_lm], 2)
        logits, possible_translation_tokens = self.output_projections[1](s_cf)
        logits = self.pre_softmax_activation(logits)
        return logits, possible_translation_tokens


class ConcatStrategy(MultiDecoderCombinationStrategy):
    """Concatenates decoder outputs."""

    def __init__(self, out_embed_dims, vocab_size, vocab_reduction_module=None):
        super().__init__(out_embed_dims, vocab_size, vocab_reduction_module)
        self.output_projection = OutputProjection(
            sum(out_embed_dims), vocab_size, vocab_reduction_module
        )

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
        select_single=None,
    ):
        assert select_single is None
        return self.output_projection(
            torch.cat(unprojected_outs, 2),
            src_tokens,
            input_tokens,
            possible_translation_tokens,
        )


class BottleneckStrategy(MultiDecoderCombinationStrategy):
    """Concatenation of decoder outputs followed by a bottleneck layer."""

    def __init__(self, out_embed_dims, vocab_size, vocab_reduction_module=None):
        super().__init__(out_embed_dims, vocab_size, vocab_reduction_module)
        dim = out_embed_dims[0]
        self.bottleneck = Linear(sum(out_embed_dims), dim)
        self.output_projection = OutputProjection(
            dim, vocab_size, vocab_reduction_module
        )

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
        select_single=None,
    ):
        assert select_single is None
        return self.output_projection(
            self.bottleneck(torch.cat(unprojected_outs, 2)),
            src_tokens,
            input_tokens,
            possible_translation_tokens,
        )


class DeepBottleneckStrategy(MultiDecoderCombinationStrategy):
    """Bottleneck strategy with an additional non-linear layer."""

    def __init__(
        self,
        out_embed_dims,
        vocab_size,
        vocab_reduction_module=None,
        activation_fn=torch.nn.ReLU,
    ):
        super().__init__(out_embed_dims, vocab_size, vocab_reduction_module)
        dim = out_embed_dims[0]
        self.bottleneck = nn.Sequential(
            Linear(sum(out_embed_dims), dim, bias=True),
            activation_fn(),
            Linear(dim, dim, bias=True),
        )
        self.output_projection = OutputProjection(
            dim, vocab_size, vocab_reduction_module
        )

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
        select_single=None,
    ):
        assert select_single is None
        return self.output_projection(
            self.bottleneck(torch.cat(unprojected_outs, 2)),
            src_tokens,
            input_tokens,
            possible_translation_tokens,
        )


class BaseWeightedStrategy(MultiDecoderCombinationStrategy):
    """Base class for strategies with explicitly learned weights."""

    def __init__(
        self,
        out_embed_dims,
        vocab_size,
        vocab_reduction_module=None,
        fixed_weights=None,
        hidden_layer_size=32,
        activation_fn=torch.nn.ReLU,
        logit_fn=torch.exp,
    ):
        """Initializes a combination strategy with explicit weights.

        Args:
            out_embed_dims (list): List of output dimensionalities of the
                decoders.
            vocab_size (int): Size of the output projection.
            vocab_reduction_module: For vocabulary reduction
            fixed_weights (list): If not None, use these fixed weights rather
                than a gating network.
            hidden_layer_size (int): Size of the hidden layer of the gating
                network.
            activation_fn: Non-linearity at the hidden layer.
            norm_fn: Function to use for normalization (exp or sigmoid).
        """
        super().__init__(out_embed_dims, vocab_size, vocab_reduction_module)
        if fixed_weights is None:
            self.fixed_weights = None
            self.gating_network = nn.Sequential(
                Linear(sum(out_embed_dims), hidden_layer_size, bias=True),
                activation_fn(),
                Linear(hidden_layer_size, len(out_embed_dims), bias=True),
            )
            self.logit_fn = logit_fn
        else:
            assert len(fixed_weights) == len(out_embed_dims)
            self.fixed_weights = maybe_cuda(torch.Tensor(fixed_weights).view(1, 1, -1))

    def compute_weights(self, unprojected_outs, select_single=None):
        """Derive interpolation weights from unprojected decoder outputs.

        Args:
            unprojected_outs: List of [batch_size, seq_len, out_embed_dim]
                tensors with unprojected decoder outputs.
            select_single: If not None, put all weighton n-th model.

        Returns:
            A [batch_size, seq_len, num_decoders] float32 tensor with
            normalized decoder interpolation weights.
        """
        if select_single is not None:
            sz = unprojected_outs[0].size()
            ret = maybe_cuda(torch.zeros((sz[0], sz[1], len(unprojected_outs))))
            ret[:, :, select_single] = 1.0
            return ret
        if self.fixed_weights is not None:
            return self.fixed_weights
        logits = self.logit_fn(self.gating_network(torch.cat(unprojected_outs, 2)))
        return torch.clamp(logits / torch.sum(logits, dim=2, keepdim=True), 0.0, 1.0)


class WeightedStrategy(BaseWeightedStrategy):
    """Weighted average of full logits."""

    def __init__(
        self,
        out_embed_dims,
        vocab_size,
        vocab_reduction_module=None,
        norm_fn=None,
        to_log=False,
        fixed_weights=None,
    ):
        super().__init__(out_embed_dims, vocab_size, fixed_weights=fixed_weights)
        assert vocab_reduction_module is None
        self.output_projections = nn.ModuleList(
            [OutputProjection(dim, vocab_size) for dim in out_embed_dims]
        )
        self.norm_fn = norm_fn
        self.n_systems = len(out_embed_dims)
        self.to_log = to_log

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
        select_single=None,
    ):
        assert possible_translation_tokens is None
        weights = self.compute_weights(unprojected_outs, select_single)
        weights = [weights[:, :, i : i + 1] for i in range(self.n_systems)]
        logits = [p(o)[0] for p, o in zip(self.output_projections, unprojected_outs)]
        avg = average_tensors(logits, weights=weights, norm_fn=self.norm_fn)
        if self.to_log:
            avg.log_()
        return avg, None


class WeightedUnprojectedStrategy(BaseWeightedStrategy):
    """Weighted average of decoder outputs, shared projection layer."""

    def __init__(self, out_embed_dims, vocab_size, vocab_reduction_module=None):
        super().__init__(out_embed_dims, vocab_size, vocab_reduction_module)
        out_embed_dim = out_embed_dims[0]
        assert all(d == out_embed_dim for d in out_embed_dims)
        self.output_projection = OutputProjection(
            out_embed_dim, vocab_size, vocab_reduction_module
        )

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
        select_single=None,
    ):
        weights = self.compute_weights(unprojected_outs, select_single)
        weights = [weights[:, :, i : i + 1] for i in range(self.n_systems)]
        averaged_unprojected = average_tensors(unprojected_outs, weights=weights)
        return self.output_projections[0](
            averaged_unprojected, src_tokens, input_tokens, possible_translation_tokens
        )


def parse_strategy_name(strategy_name, n_models):
    modifier_idx = None
    if "-" in strategy_name:
        strategy_name, strategy_modifier = strategy_name.split("-")
        if "_" in strategy_modifier:
            strategy_modifier, modifier_idx = strategy_modifier.split("_")
            modifier_idx = int(modifier_idx)
    else:
        strategy_modifier = None
    norm_fn = None
    to_log = False
    if strategy_modifier == "probspace":
        norm_fn = F.softmax
        to_log = True
    elif strategy_modifier == "logprobspace":
        norm_fn = F.log_softmax
    if modifier_idx is not None:
        norm_fn_list = [None] * n_models
        norm_fn_list[modifier_idx] = norm_fn
        norm_fn = norm_fn_list
    return strategy_name, strategy_modifier, norm_fn, to_log


def create_strategy(
    strategy_name, out_embed_dims, vocab_size, vocab_reduction_module, fixed_weights
):
    strategy_name, strategy_modifier, norm_fn, to_log = parse_strategy_name(
        strategy_name, len(out_embed_dims)
    )
    if strategy_name == "uniform":
        return UniformStrategy(
            out_embed_dims,
            vocab_size,
            vocab_reduction_module,
            norm_fn=norm_fn,
            to_log=to_log,
        )
    elif strategy_name == "weighted":
        return WeightedStrategy(
            out_embed_dims,
            vocab_size,
            vocab_reduction_module,
            norm_fn=norm_fn,
            to_log=to_log,
            fixed_weights=fixed_weights,
        )
    elif strategy_name == "unprojected":
        return UnprojectedStrategy(out_embed_dims, vocab_size, vocab_reduction_module)
    elif strategy_name == "max" and strategy_modifier == "unprojected":
        return MaxUnprojectedStrategy(
            out_embed_dims, vocab_size, vocab_reduction_module
        )
    elif strategy_name == "concat":
        return ConcatStrategy(out_embed_dims, vocab_size, vocab_reduction_module)
    elif strategy_name == "deepfusion":
        return DeepFusionStrategy(out_embed_dims, vocab_size, vocab_reduction_module)
    elif strategy_name == "coldfusion":
        return ColdFusionStrategy(out_embed_dims, vocab_size, vocab_reduction_module)
    elif strategy_name == "bottleneck":
        return BottleneckStrategy(out_embed_dims, vocab_size, vocab_reduction_module)
    elif strategy_name == "deep_bottleneck":
        return DeepBottleneckStrategy(
            out_embed_dims, vocab_size, vocab_reduction_module
        )
    elif strategy_name == "multiplicative" and strategy_modifier == "unprojected":
        return MultiplicativeUnprojectedStrategy(
            out_embed_dims, vocab_size, vocab_reduction_module
        )
    raise RuntimeError(f"Unknown combination strategy '{strategy_name}'")


class MultiDecoder(FairseqIncrementalDecoder):
    """Multi-decoder ensembles.

    Combines multiple decoders. See the `MultiDecoderCombinationStrategy`
    implementations on how the decoder outputs are combined.
    """

    def __init__(
        self,
        src_dict,
        dst_dict,
        decoders,
        combination_strategy,
        is_lm=None,
        split_encoder=False,
        vocab_reduction_params=None,
        training_schedule="complete",
        fixed_weights=None,
    ):
        """Create a new multi-decoder instance.

        Args:
            src_dict (Dictionary): Source language dictionary.
            dst_dict (Dictionary): Target language dictionary.
            decoders (list): List of DecoderWithOutputProjection.
            combination_strategy (string): Name of the combination strategy.
                Passed through to `create_strategy()`.
            is_lm (list): List of booleans determining whether the n-th
                decoder is a language model. If None, none of the decoders are
                considered an LM.
            split_encoder (bool): If true, split encoder output, each decoder
                gets its own split.
            vocab_reduction_params: For vocabular reduction.
            training_schedule (str): Training strategy.
            fixed_weights (list): None or list of floats. If specified, use
                these fixed model weights in weighted* combination strategies.
        """
        super().__init__(dst_dict)
        if is_lm is None:
            is_lm = [False] * len(decoders)
        assert not any(decoder.project_output for decoder in decoders)
        assert len(is_lm) == len(decoders)
        self.attentive_decoder_ids = [i for i, b in enumerate(is_lm) if not b]
        self.decoders_is_lm = is_lm
        self.decoders = nn.ModuleList(decoders)
        vocab_reduction_module = None
        if vocab_reduction_params:
            vocab_reduction_module = vocab_reduction.VocabReduction(
                src_dict, dst_dict, vocab_reduction_params
            )
        self.combi_strat = create_strategy(
            combination_strategy,
            [decoder.out_embed_dim for decoder in decoders],
            len(dst_dict),
            vocab_reduction_module,
            fixed_weights,
        )
        self.split_encoder = split_encoder
        self.unfreeze_single = False
        self.separate_training = False
        self.unfreeze_idx = -1
        if self.training:
            if training_schedule in ["freeze_all", "freeze_all_decoders"]:
                self.freeze_decoders()
            elif training_schedule.startswith(
                "unfreeze_dec_"
            ) or training_schedule.startswith("unfreeze_encdec_"):
                _, _, n = training_schedule.split("_")
                self.freeze_decoders(int(n))
            elif training_schedule in ["unfreeze_single", "unfreeze_single_decoder"]:
                self.unfreeze_single = True
                self.unfreeze_mod = len(decoders)
            elif training_schedule == "separate":
                self.unfreeze_single = True
                self.unfreeze_mod = len(decoders) + 1
                self.separate_training = True
            elif training_schedule != "complete":
                raise RuntimeError(f"Unknown training schedule '{training_schedule}'")

    def freeze_decoders(self, except_idx=-1):
        """Freezes weights in all decoders except `except_idx`."""
        unfreeze_nth_component(self.decoders, except_idx)
        try:
            unfreeze_nth_component(self.combi_strat.output_projections, except_idx)
        except AttributeError:
            pass  # combi_strat does not have multiple output projections

    def forward(
        self,
        input_tokens,
        encoder_out,
        incremental_state=None,
        possible_translation_tokens=None,
    ):
        if self.unfreeze_single:
            self.unfreeze_idx = (self.unfreeze_idx + 1) % self.unfreeze_mod
            if self.separate_training:
                unfreeze_combi_strat = len(self.decoders) == self.unfreeze_idx
                for p in self.combi_strat.parameters():
                    p.requires_grad = unfreeze_combi_strat
            self.freeze_decoders(self.unfreeze_idx)
        if incremental_state is None:
            incremental_state = {
                decoder_id: None for decoder_id in range(len(self.decoders))
            }
        decoder_outs = []
        decoder_contexts = self._get_contexts(encoder_out)
        for decoder_id, decoder in enumerate(self.decoders):
            if decoder_id not in incremental_state:
                incremental_state[decoder_id] = {}
            decoder_outs.append(
                decoder.forward_unprojected(
                    input_tokens,
                    decoder_contexts[decoder_id],
                    incremental_state=incremental_state[decoder_id],
                )
            )
        mean_attn_scores = average_tensors(
            [decoder_outs[decoder_id][1] for decoder_id in self.attentive_decoder_ids]
        )
        select_single = None
        if self.separate_training and not unfreeze_combi_strat:
            select_single = self.unfreeze_idx
        logits, possible_translation_tokens = self.combi_strat(
            [x for x, _ in decoder_outs],
            src_tokens=encoder_out[4],
            input_tokens=input_tokens if self.training else None,
            possible_translation_tokens=possible_translation_tokens,
            select_single=select_single,
        )
        return logits, mean_attn_scores, possible_translation_tokens

    def _get_contexts(self, encoder_out):
        encoder_outs, final_hidden, final_cell, src_lengths, src_tokens = encoder_out
        if self.split_encoder:
            split_encoder_outs = []
            offset = 0
            for decoder in self.decoders:
                next_offset = offset + decoder.encoder_hidden_dim
                split_encoder_outs.append(
                    (
                        encoder_outs[:, :, offset:next_offset],
                        final_hidden[:, :, offset:next_offset],
                        final_cell[:, :, offset:next_offset],
                        src_lengths,
                        src_tokens,
                    )
                )
                offset = next_offset
            assert offset == encoder_outs.size(2)
        else:
            split_encoder_outs = [encoder_out] * len(self.decoders)
        if any(self.decoders_is_lm):
            num_layers, bsz, _ = final_cell.size()
            ones = torch.ones((num_layers, bsz, 1)).type_as(final_cell)
            dummy_out = torch.ones((1, bsz, 1)).type_as(final_cell)
            lm_encoder_outs = dummy_out, ones, ones, src_lengths, src_tokens
            for decoder_id, is_lm in enumerate(self.decoders_is_lm):
                if is_lm:
                    split_encoder_outs[decoder_id] = lm_encoder_outs
        return split_encoder_outs

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        if not incremental_state:
            return
        for decoder_id, decoder in enumerate(self.decoders):
            decoder.reorder_incremental_state(incremental_state[decoder_id], new_order)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number


def import_individual_models(restore_files, trainer):
    param2size = {}
    for name, param in trainer.model.named_parameters():
        param2size[name] = param.size()
    cuda_device = torch.cuda.current_device()
    model_state = {}
    for idx, filename in enumerate(restore_files):
        sub_state = torch.load(
            filename,
            map_location=lambda s, l: default_restore_location(
                s, "cuda:{}".format(cuda_device)
            ),
        )
        for name, value in sub_state["model"].items():
            new_name = None
            if name.startswith("encoder."):
                subname = name[8:]
                new_name = f"encoder.encoders.{idx}.{subname}"
            elif name == "decoder.output_projection_w":
                new_name = (
                    f"decoder.combi_strat.output_projections.{idx}."
                    f"output_projection_w"
                )
            elif name == "decoder.output_projection_b":
                new_name = (
                    f"decoder.combi_strat.output_projections.{idx}."
                    f"output_projection_b"
                )
            elif name.startswith("decoder."):
                subname = name[8:]
                new_name = f"decoder.decoders.{idx}.{subname}"
            if new_name is None:
                print(f"WARN: Ignoring {name} in {filename} (no match)")
            elif new_name not in param2size:
                print(f"WARN: Could not find {new_name}. Check architectures")
            elif value.size() != param2size[new_name]:
                print(
                    f"WARN: Tried to map {name} to {new_name}, but sizes do not match "
                    f"({value.size()} != {param2size[new_name]})"
                )
            else:
                model_state[new_name] = value
    trainer.model.load_state_dict(model_state, strict=False)
    print(f"|  Imported {len(model_state)} parameters.")
    trainer._optim_history = []
