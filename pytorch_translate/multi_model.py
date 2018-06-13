#!/usr/bin/env python3

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder
from pytorch_translate.common_layers import Linear, OutputProjection
from pytorch_translate import vocab_reduction


def average_tensors(tensor_list, prob_space=False):
    stacked = torch.stack(tensor_list)
    if not prob_space:
        return torch.mean(stacked, dim=0)
    probs = F.softmax(stacked)
    return torch.log(torch.mean(probs, dim=0))


class MultiEncoder(FairseqEncoder):
    """Concatenates the outputs of multiple encoders."""

    def __init__(self, dictionary, encoders):
        super().__init__(dictionary)
        self.encoders = nn.ModuleList(encoders)

    def forward(self, src_tokens, src_lengths):
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
        self, out_embed_dims, vocab_size, vocab_reduction_module=None, prob_space=False
    ):
        super().__init__(out_embed_dims, vocab_size)
        assert vocab_reduction_module is None
        self.output_projections = nn.ModuleList(
            [OutputProjection(dim, vocab_size) for dim in out_embed_dims]
        )
        self.prob_space = prob_space

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
    ):
        assert possible_translation_tokens is None
        logits = [p(o)[0] for p, o in zip(self.output_projections, unprojected_outs)]
        return average_tensors(logits, prob_space=self.prob_space), None


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
    ):
        return self.output_projection(
            average_tensors(unprojected_outs),
            src_tokens,
            input_tokens,
            possible_translation_tokens,
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
    ):
        return self.output_projection(
            torch.prod(self.activation(torch.stack(unprojected_outs)), dim=0),
            src_tokens,
            input_tokens,
            possible_translation_tokens,
        )


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
    ):
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
    ):
        return self.output_projection(
            self.bottleneck(torch.cat(unprojected_outs, 2)),
            src_tokens,
            input_tokens,
            possible_translation_tokens,
        )


class BaseWeightedStrategy(MultiDecoderCombinationStrategy):
    """Base class for strategies with explicitly learned weights."""

    def __init__(self, out_embed_dims, vocab_size, vocab_reduction_module=None):
        super().__init__(out_embed_dims, vocab_size, vocab_reduction_module)
        self.weight_projection = Linear(
            sum(out_embed_dims), len(out_embed_dims), bias=True
        )

    def compute_weights(self, unprojected_outs):
        """Derive interpolation weights from unprojected decoder outputs.

        Args:
            unprojected_outs: List of [batch_size, seq_len, out_embed_dim]
                tensors with unprojected decoder outputs.

        Returns:
            A [batch_size, seq_len, num_decoders] float32 tensor with
            normalized decoder interpolation weights.
        """
        logits = torch.exp(self.weight_projection(torch.cat(unprojected_outs, 2)))
        return logits / torch.sum(logits, dim=2, keepdim=True)


class WeightedStrategy(BaseWeightedStrategy):
    """Weighted average of full logits."""

    def __init__(
        self, out_embed_dims, vocab_size, vocab_reduction_module=None, prob_space=False
    ):
        super().__init__(out_embed_dims, vocab_size)
        assert vocab_reduction_module is None
        self.output_projections = nn.ModuleList(
            [OutputProjection(dim, vocab_size) for dim in out_embed_dims]
        )
        self.prob_space = prob_space

    def forward(
        self,
        unprojected_outs,
        src_tokens=None,
        input_tokens=None,
        possible_translation_tokens=None,
    ):
        assert possible_translation_tokens is None
        weights = self.compute_weights(unprojected_outs)
        logits = [
            weights[:, :, i : i + 1] * p(o)[0]
            for i, (p, o) in enumerate(zip(self.output_projections, unprojected_outs))
        ]
        return average_tensors(logits, prob_space=self.prob_space), None


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
    ):
        weights = self.compute_weights(unprojected_outs)
        averaged_unprojected = average_tensors(
            [weights[:, :, i : i + 1] * o for i, o in enumerate(unprojected_outs)]
        )
        return self.output_projection(
            averaged_unprojected, src_tokens, input_tokens, possible_translation_tokens
        )


def create_strategy(strategy_name, out_embed_dims, vocab_size, vocab_reduction_module):
    if strategy_name == "uniform":
        return UniformStrategy(out_embed_dims, vocab_size, vocab_reduction_module)
    elif strategy_name == "uniform-probspace":
        return UniformStrategy(
            out_embed_dims, vocab_size, vocab_reduction_module, prob_space=True
        )
    elif strategy_name == "unprojected":
        return UnprojectedStrategy(out_embed_dims, vocab_size, vocab_reduction_module)
    elif strategy_name == "weighted":
        return WeightedStrategy(out_embed_dims, vocab_size, vocab_reduction_module)
    elif strategy_name == "weighted-probspace":
        return WeightedStrategy(
            out_embed_dims, vocab_size, vocab_reduction_module, prob_space=True
        )
    elif strategy_name == "weighted-unprojected":
        return WeightedUnprojectedStrategy(
            out_embed_dims, vocab_size, vocab_reduction_module
        )
    elif strategy_name == "concat":
        return ConcatStrategy(out_embed_dims, vocab_size, vocab_reduction_module)
    elif strategy_name == "bottleneck":
        return BottleneckStrategy(out_embed_dims, vocab_size, vocab_reduction_module)
    elif strategy_name == "multiplicative-unprojected":
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
        split_encoder=False,
        vocab_reduction_params=None,
    ):
        """Create a new multi-decoder instance.

        Args:
            src_dict (Dictionary): Source language dictionary.
            dst_dict (Dictionary): Target language dictionary.
            decoders (list): List of DecoderWithOutputProjection.
            combination_strategy (string): Name of the combination strategy.
                Passed through to `create_strategy()`.
            split_encoder (bool): If true, split encoder output, each decoder
                gets its own split.
            vocab_reduction_params: For vocabular reduction.
        """
        super().__init__(dst_dict)
        assert not any(decoder.project_output for decoder in decoders)
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
        )
        self.split_encoder = split_encoder

    def forward(
        self,
        input_tokens,
        encoder_out,
        incremental_state=None,
        possible_translation_tokens=None,
    ):
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
            [attn_scores for _, attn_scores in decoder_outs if attn_scores is not None]
        )
        logits, possible_translation_tokens = self.combi_strat(
            [x for x, _ in decoder_outs],
            src_tokens=encoder_out[4],
            input_tokens=input_tokens if self.training else None,
            possible_translation_tokens=possible_translation_tokens,
        )
        return logits, mean_attn_scores, possible_translation_tokens

    def _get_contexts(self, encoder_out):
        if self.split_encoder:
            (
                encoder_outs,
                final_hidden,
                final_cell,
                src_lengths,
                src_tokens,
            ) = encoder_out
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
            return split_encoder_outs
        return [encoder_out] * len(self.decoders)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        if not incremental_state:
            return
        for decoder_id, decoder in enumerate(self.decoders):
            decoder.reorder_incremental_state(incremental_state[decoder_id], new_order)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number
