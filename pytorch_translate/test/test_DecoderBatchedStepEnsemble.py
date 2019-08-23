#!/usr/bin/env python3

import unittest

import numpy as np
import torch
from pytorch_translate.beam_search_and_decode_v2 import (
    DecoderBatchedStepEnsemble2BeamWithEOS,
)
from pytorch_translate.ensemble_export import EncoderEnsemble
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestDecoderBatchedStepEnsemble(unittest.TestCase):
    def test_decoder_ensemble_2beam_with_eos(self):
        """
        Test the functionality of DecoderBatchedStepEnsemble2BeamWithEOS class.
        At final step, it generates EOS tokens.
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
        decoder_ensemble_2beam_with_eos = DecoderBatchedStepEnsemble2BeamWithEOS(
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
        active_hypos = torch.LongTensor([0])
        num_steps = torch.LongTensor([maxLen])

        decoder_with_eos_first_step_outputs = decoder_ensemble_2beam_with_eos(
            prev_tokens,
            prev_scores,
            active_hypos,
            timestep,
            final_step,
            *encoder_outputs,
        )

        (
            cand_tokens,
            cand_scores,
            cand_prev_hypos_indices,
            cand_attn_weights,
            active_hypos,
            *states_with_eos,
        ) = decoder_with_eos_first_step_outputs

        for i in range(len([model])):
            states_with_eos[i] = states_with_eos[i].repeat(1, beam_size, 1)

        for i in range(num_steps - 1):
            decoder_step_with_eos_outputs = decoder_ensemble_2beam_with_eos(
                cand_tokens,
                cand_scores,
                active_hypos,
                torch.tensor([i + 1]),
                final_step,
                *states_with_eos,
            )
            (
                cand_tokens,
                cand_scores,
                cand_prev_hypos_indices,
                cand_attn_weights,
                active_hypos,
                *states_with_eos,
            ) = decoder_step_with_eos_outputs

        # Test the outputs of final tesp
        decoder_final_with_eos_outputs = decoder_ensemble_2beam_with_eos(
            cand_tokens,
            cand_scores,
            active_hypos,
            torch.tensor([num_steps]),
            torch.tensor([True]),
            *states_with_eos,
        )

        np.testing.assert_array_equal(
            decoder_final_with_eos_outputs[0],
            torch.LongTensor([eos_token]).repeat(2 * beam_size),
        )
        np.testing.assert_array_equal(
            decoder_final_with_eos_outputs[2],
            torch.LongTensor(np.array([i for i in range(2 * beam_size)])),
        )
