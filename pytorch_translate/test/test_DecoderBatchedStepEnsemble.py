#!/usr/bin/env python3

import unittest

import numpy as np
import torch
from pytorch_translate.ensemble_export import (
    DecoderBatchedStepEnsemble,
    DecoderBatchedStepEnsembleWithEOS,
    EncoderEnsemble,
)
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestDecoderBatchedStepEnsemble(unittest.TestCase):
    def test_decoder_ensemble_with_eos(self):
        """
        This is to test the functionality of DecoderBatchedStepEnsembleWithEOS class.
        We expect it generates same outputs with DecoderBatchedStepEnsemble before
        final step. At final step, it generates EOS tokens.
        """
        test_args = test_utils.ModelParamsDict(arch="rnn")
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        model = task.build_model(test_args)
        eos_token = tgt_dict.eos()

        encoder_ensemble = EncoderEnsemble([model])
        src_tokens = torch.LongTensor([4, 5, 6, 7, 8]).unsqueeze(1)
        src_lengths = torch.LongTensor([5])
        enc_inputs = (src_tokens, src_lengths)
        encoder_outputs = encoder_ensemble(*enc_inputs)

        beam_size = 8
        word_reward = 1
        unk_reward = -1
        decoder_ensemble = DecoderBatchedStepEnsemble(
            models=[model],
            tgt_dict=tgt_dict,
            beam_size=beam_size,
            word_reward=word_reward,
            unk_reward=unk_reward,
        )
        decoder_ensemble_with_eos = DecoderBatchedStepEnsembleWithEOS(
            models=[model],
            tgt_dict=tgt_dict,
            beam_size=beam_size,
            word_reward=word_reward,
            unk_reward=unk_reward,
        )

        prev_tokens = torch.LongTensor([eos_token])
        prev_scores = torch.FloatTensor([0.0])
        timestep = torch.LongTensor([0])
        final_step = torch.tensor([False], dtype=torch.bool)
        maxLen = 5
        num_steps = torch.LongTensor([maxLen])

        decoder_first_step_outputs = decoder_ensemble(
            prev_tokens, prev_scores, timestep, *encoder_outputs
        )

        decoder_with_eos_first_step_outputs = decoder_ensemble_with_eos(
            prev_tokens, prev_scores, timestep, final_step, *encoder_outputs
        )

        # Test results at first step
        self._test_base(decoder_first_step_outputs, decoder_with_eos_first_step_outputs)

        (
            prev_tokens,
            prev_scores,
            prev_hypos_indices,
            attn_weights,
            *states,
        ) = decoder_first_step_outputs

        # Tile is needed after first step
        for i in range(len([model])):
            states[i] = states[i].repeat(1, beam_size, 1)

        (
            prev_tokens_with_eos,
            prev_scores_with_eos,
            prev_hypos_indices_with_eos,
            attn_weights_with_eos,
            *states_with_eos,
        ) = decoder_with_eos_first_step_outputs

        for i in range(len([model])):
            states_with_eos[i] = states_with_eos[i].repeat(1, beam_size, 1)

        for i in range(num_steps - 1):
            decoder_step_outputs = decoder_ensemble(
                prev_tokens, prev_scores, torch.tensor([i + 1]), *states
            )
            (
                prev_tokens,
                prev_scores,
                prev_hypos_indices,
                attn_weights,
                *states,
            ) = decoder_step_outputs
            decoder_step_with_eos_outputs = decoder_ensemble_with_eos(
                prev_tokens_with_eos,
                prev_scores_with_eos,
                torch.tensor([i + 1]),
                final_step,
                *states_with_eos,
            )
            (
                prev_tokens_with_eos,
                prev_scores_with_eos,
                prev_hypos_indices_with_eos,
                attn_weights_with_eos,
                *states_with_eos,
            ) = decoder_step_with_eos_outputs

            # Test results at each step
            self._test_base(decoder_step_outputs, decoder_step_with_eos_outputs)

        # Test the outputs of final tesp
        decoder_final_with_eos_outputs = decoder_ensemble_with_eos(
            prev_tokens_with_eos,
            prev_scores_with_eos,
            torch.tensor([num_steps]),
            torch.tensor([True]),
            *states_with_eos,
        )

        np.testing.assert_array_equal(
            decoder_final_with_eos_outputs[0],
            torch.LongTensor([eos_token]).repeat(beam_size),
        )
        np.testing.assert_array_equal(
            decoder_final_with_eos_outputs[2],
            torch.LongTensor(np.array([i for i in range(beam_size)])),
        )

    def _test_base(self, decoder_outputs, decoder_with_eos_outputs):

        # Compare output tokens at each step
        np.testing.assert_array_equal(
            decoder_outputs[0].tolist(), decoder_with_eos_outputs[0].tolist()
        )
        # Compare output scores at each step
        np.testing.assert_array_almost_equal(
            decoder_outputs[1].tolist(), decoder_with_eos_outputs[1].tolist(), decimal=1
        )
        # Compare output prev_hypos_indices at each step
        np.testing.assert_array_equal(
            decoder_outputs[2].tolist(), decoder_with_eos_outputs[2].tolist()
        )
        # Compare output attn_weights at each step
        np.testing.assert_array_almost_equal(
            decoder_outputs[3].detach().numpy(),
            decoder_with_eos_outputs[3].detach().numpy(),
        )
