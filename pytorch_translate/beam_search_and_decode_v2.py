#!/usr/bin/env python3

from typing import List, Tuple

import numpy as np
import torch
import torch.jit
import torch.jit.quantized
from pytorch_translate.beam_decode import BeamDecode
from pytorch_translate.ensemble_export import (
    DecoderBatchedStepEnsemble,
    EncoderEnsemble,
    FakeCharSourceEncoderEnsemble,
    load_models_from_checkpoints,
)
from torch import Tensor


class DecoderBatchedStepEnsemble2BeamWithEOS(DecoderBatchedStepEnsemble):
    """
    This class inherits DecoderBatchedStepEnsemble class. While keeping the basic
    functionality of running decoding ensemble, two new features are added:
    expanding double beam size at each search step in case half are eos, appending
    extra EOS tokens at the end.
    """

    # TODO:this class will be merged with upstream after BeamSearchAndDecodeV1 shipped.
    def forward(
        self,
        input_tokens,
        prev_scores,
        active_hypos,
        timestep,
        final_step,
        *inputs,
        src_tuple=None,
    ):
        # input_tokens size: 2 * beam_size -> beam_size,
        # since we only need half of them which are active.
        input_tokens = input_tokens.index_select(dim=0, index=active_hypos).unsqueeze(1)
        prev_scores = prev_scores.index_select(dim=0, index=active_hypos)

        eos_token = torch.LongTensor([self.tgt_dict.eos()])

        (
            log_probs_per_model,
            attn_weights_per_model,
            state_outputs,
            beam_axis_per_state,
            possible_translation_tokens,
        ) = self._get_decoder_outputs(
            input_tokens, prev_scores, timestep, *inputs, src_tuple=src_tuple
        )

        average_log_probs = torch.mean(
            torch.cat(log_probs_per_model, dim=1), dim=1, keepdim=True
        )

        if possible_translation_tokens is None:
            word_rewards = self.word_rewards
        else:
            word_rewards = self.word_rewards.index_select(
                0, possible_translation_tokens
            )
        word_rewards = word_rewards.unsqueeze(dim=0).unsqueeze(dim=0)

        average_log_probs_with_rewards = average_log_probs + word_rewards

        average_attn_weights = torch.mean(
            torch.cat(attn_weights_per_model, dim=1), dim=1, keepdim=True
        )

        # need control-flow to see if it's final_step, thus written in script.
        @torch.jit.script
        def generate_outputs(
            final_step: Tensor,
            average_log_probs_with_rewards: Tensor,
            average_attn_weights: Tensor,
            prev_scores: Tensor,
            eos_token: Tensor,
            beam_size: int,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            # expand 2 * beam_size in case half of them are eos tokens
            double_beam_size = 2 * beam_size

            if bool(final_step):

                # at final step, we just select eos token and its corresponding score
                # as best_tokens and eos_scores.
                cand_tokens = eos_token.repeat(double_beam_size)
                # eos_scores size: (beam_size, 1, 1)
                eos_scores = average_log_probs_with_rewards.index_select(
                    dim=2, index=eos_token
                )
                eos_scores_flat = eos_scores.view(-1)
                cand_scores = prev_scores.view(-1) + eos_scores_flat
                # cand_scores size: beam_size -> 2 * beam_size
                cand_scores = cand_scores.repeat(2)
                cand_prev_hypos = torch.arange(0, double_beam_size).type_as(cand_tokens)
                cand_attention_weights = average_attn_weights.squeeze(1).repeat(2, 1)
                # active_hypos size: beam_size
                active_hypos = torch.arange(0, beam_size).type_as(cand_tokens)
            else:

                # Here we keep consistent with SequenceGenerator, take 2*beam_size best
                # predictions per step, will select top beam_size of these which don't
                # predict eos to continue with.
                cand_scores_k_by_2k, cand_tokens_k_by_2k = torch.topk(
                    average_log_probs_with_rewards.squeeze(1), k=double_beam_size
                )

                prev_scores_k_by_2k = prev_scores.view(-1, 1).expand(
                    -1, double_beam_size
                )
                # size is (beam_size, 2 * beam_size)
                total_scores_k_by_2k = cand_scores_k_by_2k + prev_scores_k_by_2k

                total_scores_flat_2k = total_scores_k_by_2k.view(-1)
                # size is (beam_size * 2 * beam_size)
                cand_tokens_flat_2k = cand_tokens_k_by_2k.view(-1)
                # size is (2 * beam_size)
                cand_scores, cand_indices = torch.topk(
                    total_scores_flat_2k, k=double_beam_size
                )
                # size is (2 * beam_size)
                cand_tokens = cand_tokens_flat_2k.index_select(
                    dim=0, index=cand_indices
                ).view(-1)

                # size is (2 * beam_size)
                eos_mask = cand_tokens.eq(eos_token[0])
                cand_prev_hypos = cand_indices // double_beam_size
                cand_prev_hypos = cand_prev_hypos.type_as(cand_tokens)

                cand_offsets = torch.arange(0, double_beam_size)

                active_mask = torch.add(
                    eos_mask.type_as(cand_offsets) * double_beam_size, cand_offsets
                )
                # select active hypos, size is (beam_size)
                _, active_hypos = torch.topk(
                    active_mask, k=beam_size, dim=0, largest=False, sorted=True
                )

                cand_attention_weights = average_attn_weights.index_select(
                    dim=0, index=cand_prev_hypos
                ).squeeze(1)

            return (
                cand_tokens,
                cand_scores,
                cand_prev_hypos,
                cand_attention_weights,
                active_hypos,
            )

        (
            cand_tokens,
            cand_scores,
            cand_prev_hypos,
            cand_attention_weights,
            active_hypos,
        ) = generate_outputs(
            final_step,
            average_log_probs_with_rewards,
            average_attn_weights,
            prev_scores,
            eos_token=eos_token,
            beam_size=self.beam_size,
        )

        # select active prev_hypos
        active_prev_hypos = cand_prev_hypos.index_select(dim=0, index=active_hypos)
        if possible_translation_tokens is not None:
            cand_tokens = possible_translation_tokens.index_select(
                dim=0, index=cand_tokens
            )

        self.input_names = ["prev_tokens", "prev_scores", "active_hypos", "timestep"]
        for i in range(len(self.models)):
            self.input_names.append(f"fixed_input_{i}")

        if possible_translation_tokens is not None:
            self.input_names.append("possible_translation_tokens")

        active_outputs = [
            cand_tokens,
            cand_scores,
            cand_prev_hypos,
            cand_attention_weights,
            active_hypos,
        ]
        self.output_names = [
            "cand_tokens",
            "cand_scores",
            "cand_prev_hypos",
            "cand_attention_weights",
            "active_hypos",
        ]
        for i in range(len(self.models)):
            self.output_names.append(f"fixed_input_{i}")
            if self.tile_internal:
                active_outputs.append(inputs[i].repeat(1, self.beam_size, 1))
            else:
                active_outputs.append(inputs[i])

        if possible_translation_tokens is not None:
            self.output_names.append("possible_translation_tokens")
            active_outputs.append(possible_translation_tokens)

        # just keep states for active_hypos
        for i, state in enumerate(state_outputs):
            beam_axis = beam_axis_per_state[i]
            if beam_axis is None:
                next_state = state
            else:
                next_state = state.index_select(dim=beam_axis, index=active_prev_hypos)
            active_outputs.append(next_state)
            self.output_names.append(f"state_output_{i}")
            self.input_names.append(f"state_input_{i}")

        return tuple(active_outputs)


class BeamDecodeWithEOS(BeamDecode):
    """
    Run beam decoding based on the beam search output from
    DecoderBatchedStepEnsemble2BeamWithEOS. The differences compared with BeamDecode is:
    1.there's no need to check prev_hypos finished or not when trying to get all end
    states since we don't expand at eos token in DecoderBatchedStepEnsemble2BeamWithEOS.
    2. add extra step for eos token at the end.
    """

    # TODO: (lizguo) This class will be merged with upstream later.
    @torch.jit.script_method
    def _get_all_end_states(
        self,
        beam_tokens: Tensor,
        beam_scores: Tensor,
        beam_prev_indices: Tensor,
        num_steps: int,
    ) -> Tensor:
        min_score = float("inf")
        min_index = -1
        end_states = torch.jit.annotate(List[Tensor], [])

        position = 1
        while bool(position <= num_steps + 1):
            for hyp_index in range(self.beam_size):
                if bool(beam_tokens[position][hyp_index] == self.eos_token_id) or bool(
                    position == num_steps + 1
                ):
                    hypo_score = float(beam_scores[position][hyp_index])
                    if bool(self.length_penalty != 0):
                        hypo_score = hypo_score / float(position) ** float(
                            self.length_penalty
                        )
                    end_states, min_score, min_index = self._add_to_end_states(
                        end_states,
                        min_score,
                        torch.tensor([hypo_score, float(position), float(hyp_index)]),
                        min_index,
                    )
            position = position + 1

        end_states = torch.stack(end_states)

        _, sorted_end_state_indices = end_states[:, 0].sort(dim=0, descending=True)
        end_states = end_states[sorted_end_state_indices, :]
        return end_states

    @torch.jit.script_method
    def _check_dimensions(
        self,
        beam_tokens: Tensor,
        beam_scores: Tensor,
        token_weights: Tensor,
        beam_prev_indices: Tensor,
        num_steps: int,
    ) -> None:

        assert (
            beam_tokens.size(1) == 2 * self.beam_size
        ), "Dimension of beam_tokens : {} and beam size : {} are not consistent".format(
            beam_tokens.size(), self.beam_size
        )
        assert beam_scores.size(1) == 2 * self.beam_size, (
            "Dimension of beam_scores : {} and beam size : {} "
            "are not consistent".format(beam_scores.size(), self.beam_size)
        )
        assert token_weights.size(1) == 2 * self.beam_size, (
            "Dimension of token_weights : {} and beam size : {} "
            "are not consistent".format(token_weights.size(), self.beam_size)
        )
        assert (
            beam_prev_indices.size(1) == 2 * self.beam_size
        ), "Dimension of beam_prev_indices : {} and beam size : {} "
        "are not consistent".format(beam_prev_indices.size(), self.beam_size)

        assert beam_tokens.size(0) <= num_steps + 2, (
            "Dimension of beam_tokens : {} and num_steps : {} "
            "are not consistent".format(beam_tokens.size(), num_steps)
        )
        assert beam_scores.size(0) <= num_steps + 2, (
            "Dimension of beam_scores : {} and num_steps : {} "
            "are not consistent".format(beam_scores.size(), num_steps)
        )
        assert token_weights.size(0) <= num_steps + 2, (
            "Dimension of token_weights : {} and num_steps : {} "
            "are not consistent".format(token_weights.size(), num_steps)
        )
        assert beam_prev_indices.size(0) <= num_steps + 2, (
            "Dimension of beam_prev_indices : {} and num_steps : {} "
            "are not consistent".format(beam_prev_indices.size(), num_steps)
        )


class BeamSearchAndDecodeV2(torch.jit.ScriptModule):
    """
    The difference between BeamSearchAndDecodeV2 and BeamSearchAndDecode is: V2 calls
    DecoderBatchedStepEnsemble2BeamWithEOS instead of DecoderBatchedStepEnsemble when
    running beam search. Also, since extra EOS token has been added, it calls
    BeamDecodeWithEOS when running beam decoding which supports adding extra EOS token.
    """

    def __init__(
        self,
        models,
        tgt_dict,
        src_tokens,
        src_lengths,
        eos_token_id,
        length_penalty,
        nbest,
        beam_size,
        stop_at_eos,
        word_reward=0,
        unk_reward=0,
        quantize=False,
    ):
        super().__init__()

        self.models = models
        self.tgt_dict = tgt_dict
        self.beam_size = torch.jit.Attribute(beam_size, int)
        self.word_reward = torch.jit.Attribute(word_reward, float)
        self.unk_reward = torch.jit.Attribute(unk_reward, float)

        encoder_ens = EncoderEnsemble(self.models)
        encoder_ens.enable_precompute_reduced_weights = True

        if quantize:
            encoder_ens = torch.jit.quantized.quantize_linear_modules(encoder_ens)
            encoder_ens = torch.jit.quantized.quantize_rnn_cell_modules(encoder_ens)

        # not support char source model
        self.is_char_source = False
        enc_inputs = (src_tokens, src_lengths)
        example_encoder_outs = encoder_ens(*enc_inputs)
        self.encoder_ens = torch.jit.trace(
            encoder_ens, enc_inputs, _force_outplace=True
        )
        self.encoder_ens_char_source = FakeCharSourceEncoderEnsemble()

        decoder_ens = DecoderBatchedStepEnsemble2BeamWithEOS(
            self.models,
            tgt_dict,
            beam_size,
            word_reward,
            unk_reward,
            tile_internal=False,
        )
        decoder_ens.enable_precompute_reduced_weights = True
        if quantize:
            decoder_ens = torch.jit.quantized.quantize_linear_modules(decoder_ens)
            decoder_ens = torch.jit.quantized.quantize_rnn_cell_modules(decoder_ens)
            decoder_ens = torch.jit.quantized.quantize_rnn_modules(decoder_ens)
        decoder_ens_tile = DecoderBatchedStepEnsemble2BeamWithEOS(
            self.models,
            tgt_dict,
            beam_size,
            word_reward,
            unk_reward,
            tile_internal=True,
        )
        decoder_ens_tile.enable_precompute_reduced_weights = True
        if quantize:
            decoder_ens_tile = torch.jit.quantized.quantize_linear_modules(
                decoder_ens_tile
            )
            decoder_ens_tile = torch.jit.quantized.quantize_rnn_cell_modules(
                decoder_ens_tile
            )
            decoder_ens_tile = torch.jit.quantized.quantize_rnn_modules(
                decoder_ens_tile
            )
        prev_token = torch.LongTensor([0])
        prev_scores = torch.FloatTensor([0.0])
        ts = torch.LongTensor([0])
        final_step = torch.tensor([False], dtype=torch.bool)
        active_hypos = torch.LongTensor([0])

        _, _, _, _, _, *tiled_states = decoder_ens_tile(
            prev_token, prev_scores, active_hypos, ts, final_step, *example_encoder_outs
        )

        self.decoder_ens_tile = torch.jit.trace(
            decoder_ens_tile,
            (
                prev_token,
                prev_scores,
                active_hypos,
                ts,
                final_step,
                *example_encoder_outs,
            ),
            _force_outplace=True,
        )
        self.decoder_ens = torch.jit.trace(
            decoder_ens,
            (
                prev_token.repeat(self.beam_size),
                prev_scores.repeat(self.beam_size),
                active_hypos.repeat(self.beam_size),
                ts,
                final_step,
                *tiled_states,
            ),
            _force_outplace=True,
        )

        self.beam_decode = BeamDecodeWithEOS(
            eos_token_id, length_penalty, nbest, beam_size, stop_at_eos
        )

        self.input_names = [
            "src_tokens",
            "src_lengths",
            "prev_token",
            "prev_scores",
            "attn_weights",
            "prev_hypos_indices",
            "num_steps",
        ]
        self.output_names = [
            "beam_output",
            "hypothesis_score",
            "token_level_scores",
            "back_alignment_weights",
            "best_indices",
        ]

    @torch.jit.script_method
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_token: torch.Tensor,
        prev_scores: torch.Tensor,
        attn_weights: torch.Tensor,
        prev_hypos_indices: torch.Tensor,
        active_hypos: torch.Tensor,
        num_steps: int,
    ) -> List[Tuple[Tensor, float, List[float], Tensor, Tensor]]:

        enc_states = self.encoder_ens(src_tokens, src_lengths)

        # enc_states ends up being optional because of the above branch, one
        # side returns None. We should never take the path that returns None
        # so we unrap the optional type here.
        enc_states = torch.jit._unwrap_optional(enc_states)

        # remove torch.cat, keep things in a list
        all_tokens = [prev_token.repeat(repeats=[2 * self.beam_size])]
        all_scores = [prev_scores.repeat(repeats=[2 * self.beam_size])]
        all_weights = [
            attn_weights.unsqueeze(dim=0).repeat(repeats=[2 * self.beam_size, 1])
        ]
        all_prev_indices = [prev_hypos_indices]

        (
            prev_token,
            prev_scores,
            prev_hypos_indices,
            attn_weights,
            active_hypos,
            *states,
        ) = self.decoder_ens_tile(
            prev_token,
            prev_scores,
            active_hypos,
            torch.tensor([0]),
            torch.tensor([False]),
            *enc_states,  # noqa
        )
        all_tokens = all_tokens.append(prev_token)
        all_scores = all_scores.append(prev_scores)
        all_weights = all_weights.append(attn_weights)
        all_prev_indices = all_prev_indices.append(prev_hypos_indices)

        for i in range(num_steps - 1):
            (
                prev_token,
                prev_scores,
                prev_hypos_indices,
                attn_weights,
                active_hypos,
                *states,
            ) = self.decoder_ens(
                prev_token,
                prev_scores,
                active_hypos,
                torch.tensor([i + 1]),
                torch.tensor([False]),
                *states,  # noqa
            )

            all_tokens = all_tokens.append(prev_token)
            all_scores = all_scores.append(prev_scores)
            all_weights = all_weights.append(attn_weights)
            all_prev_indices = all_prev_indices.append(prev_hypos_indices)

        # add eos token as extra step
        (
            prev_token,
            prev_scores,
            prev_hypos_indices,
            attn_weights,
            active_hypos,
            *states,
        ) = self.decoder_ens(
            prev_token,
            prev_scores,
            active_hypos,
            torch.tensor([num_steps]),
            torch.tensor([True]),
            *states,
        )

        all_tokens = all_tokens.append(prev_token)
        all_scores = all_scores.append(prev_scores)
        all_weights = all_weights.append(attn_weights)
        all_prev_indices = all_prev_indices.append(prev_hypos_indices)

        outputs = torch.jit.annotate(
            List[Tuple[Tensor, float, List[float], Tensor, Tensor]], []
        )
        outputs = self.beam_decode(
            torch.stack(all_tokens, dim=0),
            torch.stack(all_scores, dim=0),
            torch.stack(all_weights, dim=0),
            torch.stack(all_prev_indices, dim=0),
            num_steps,
        )

        return outputs

    @classmethod
    def build_from_checkpoints(
        cls,
        checkpoint_filenames,
        src_dict_filename,
        dst_dict_filename,
        beam_size,
        length_penalty,
        nbest,
        word_reward=0,
        unk_reward=0,
        lexical_dict_paths=None,
    ):
        length = 10
        models, _, tgt_dict = load_models_from_checkpoints(
            checkpoint_filenames,
            src_dict_filename,
            dst_dict_filename,
            lexical_dict_paths,
        )
        src_tokens = torch.LongTensor(np.ones((length, 1), dtype="int64"))
        src_lengths = torch.IntTensor(np.array([length], dtype="int32"))
        eos_token_id = tgt_dict.eos()

        return cls(
            models,
            tgt_dict,
            src_tokens,
            src_lengths,
            eos_token_id,
            length_penalty=length_penalty,
            nbest=nbest,
            beam_size=beam_size,
            stop_at_eos=True,
            word_reward=word_reward,
            unk_reward=unk_reward,
            quantize=True,
        )

    def save_to_pytorch(self, output_path):
        def pack(s):
            if hasattr(s, "_pack"):
                s._pack()

        def unpack(s):
            if hasattr(s, "_unpack"):
                s._unpack()

        self.apply(pack)
        torch.jit.save(self, output_path)
        self.apply(unpack)
