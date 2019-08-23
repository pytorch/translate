#!/usr/bin/env python3

from typing import Tuple

import torch
import torch.jit
import torch.jit.quantized
from pytorch_translate.ensemble_export import DecoderBatchedStepEnsemble
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
                cand_prev_hypos = cand_indices / double_beam_size
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
