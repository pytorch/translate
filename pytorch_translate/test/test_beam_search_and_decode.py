#!/usr/bin/env python3

import unittest

import numpy as np
import torch
from pytorch_translate import rnn  # noqa
from pytorch_translate.beam_decode import BeamDecode, SequenceGenerator
from pytorch_translate.ensemble_export import BeamSearchAndDecode
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestBeamSearchAndDecode(unittest.TestCase):
    def test_beam_search_and_decode_generate(self):
        """
        A basic test that the output given by BeamSearchAndDecode class
        is the same as SequenceGenerator
        """
        test_args = test_utils.ModelParamsDict(arch="rnn")
        test_args.sequence_lstm = True
        BEAM_SIZE = 1
        WORD_REWARD = 1
        UNK_REWARD = -1
        LENGTH_PENALTY = 0

        PLACEHOLDER_SEQ_LENGTH = 5
        NBEST = 2
        MAX_SEQ_LEN = 7

        src_tokens = torch.LongTensor([[0, 0, 0]])
        src_lengths = torch.LongTensor([3])

        # Build model list
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)

        models = task.build_model(test_args)

        # Placeholder inputs for BeamSearchAndDecode
        placeholder_src_tokens = torch.LongTensor(
            np.ones((PLACEHOLDER_SEQ_LENGTH, 1), dtype="int64")
        )
        placeholder_src_lengths = torch.IntTensor(
            np.array([PLACEHOLDER_SEQ_LENGTH], dtype="int32")
        )
        prev_token = torch.LongTensor([tgt_dict.eos()])
        prev_scores = torch.FloatTensor([0.0])
        attn_weights = torch.zeros(src_lengths[0].item())
        prev_hypos_indices = torch.zeros(BEAM_SIZE, dtype=torch.int64)
        num_steps = torch.LongTensor([MAX_SEQ_LEN])

        # Generate output using SequenceGenerator
        translator = SequenceGenerator(
            [models],
            task.target_dictionary,
            beam_size=BEAM_SIZE,
            word_reward=WORD_REWARD,
            unk_reward=UNK_REWARD,
        )

        encoder_input = {"src_tokens": src_tokens, "src_lengths": src_lengths}
        top_seq_gen_hypothesis = translator.generate(
            encoder_input, beam_size=BEAM_SIZE, maxlen=MAX_SEQ_LEN
        )[0]

        # Generate output using BeamSearch/BeamDecode
        placeholder_src_tokens = torch.LongTensor(
            np.ones((PLACEHOLDER_SEQ_LENGTH, 1), dtype="int64")
        )
        placeholder_src_lengths = torch.IntTensor(
            np.array([PLACEHOLDER_SEQ_LENGTH], dtype="int32")
        )

        # Generate output using BeamSearchAndDecode class
        beam_search_and_decode = BeamSearchAndDecode(
            [models],
            tgt_dict=tgt_dict,
            src_tokens=placeholder_src_tokens,
            src_lengths=placeholder_src_lengths,
            eos_token_id=tgt_dict.eos(),
            length_penalty=LENGTH_PENALTY,
            nbest=NBEST,
            beam_size=BEAM_SIZE,
            stop_at_eos=True,
            word_reward=WORD_REWARD,
            unk_reward=UNK_REWARD,
            quantize=True,
        )
        beam_search_and_decode_output = beam_search_and_decode(
            src_tokens.transpose(0, 1),
            src_lengths,
            prev_token,
            prev_scores,
            attn_weights,
            prev_hypos_indices,
            num_steps[0],
        )

        for hyp_index in range(
            min(len(beam_search_and_decode_output), len(top_seq_gen_hypothesis))
        ):
            beam_search_and_decode_hypothesis = beam_search_and_decode_output[hyp_index]

            # Compare two outputs
            # We always look only from 0 to MAX_SEQ_LEN, because sequence generator
            # adds an EOS at the end after MAX_SEQ_LEN

            # Compare two hypotheses
            np.testing.assert_array_equal(
                top_seq_gen_hypothesis[hyp_index]["tokens"].tolist()[0:MAX_SEQ_LEN],
                beam_search_and_decode_hypothesis[0].tolist()[0:MAX_SEQ_LEN],
            )
            # Compare token level scores
            np.testing.assert_array_almost_equal(
                top_seq_gen_hypothesis[hyp_index]["positional_scores"].tolist()[
                    0:MAX_SEQ_LEN
                ],
                beam_search_and_decode_hypothesis[2][0:MAX_SEQ_LEN],
                decimal=1,
            )

            # Compare attention weights
            np.testing.assert_array_almost_equal(
                top_seq_gen_hypothesis[hyp_index]["attention"].numpy()[
                    :, 0:MAX_SEQ_LEN
                ],
                beam_search_and_decode_hypothesis[3].numpy()[:, 0:MAX_SEQ_LEN],
                decimal=1,
            )
            # Not testing the hypothesis score as sequence generator is adding EOS
            # at the end, it changes the final score

    def test_get_all_end_states(self):
        """
        test the function of keeping nBest end_states.
        In this example, all end_states should be: ([-4.6246, 1, 1](hit eos),
        [-7.0106, 2, 2](hit eos), [-8.5362, 3, 0](hit maxLen),
        [-9.4792, 3, 1](hit maxLen), [-10.4673, 3, 2](hit maxLen)),
        nBest should be the first three.
        """
        beam_tokens = torch.tensor([[2, 2, 2], [3, 2, 4], [1, 5, 2], [6, 6, 3]])
        beam_scores = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000],
                [-3.6237, -4.6246, -4.6427],
                [-6.5691, -6.9823, -7.0106],
                [-8.5362, -9.4792, -10.4673],
            ]
        )
        beam_prev_indices = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 2], [1, 0, 1]])

        beam_decode = BeamDecode(
            eos_token_id=2, length_penalty=0.0, nbest=3, beam_size=3, stop_at_eos=True
        )

        all_end_states = beam_decode._get_all_end_states(
            beam_tokens, beam_scores, beam_prev_indices, num_steps=3
        )
        desired_end_states_1 = torch.tensor([-4.6246, 1, 1])
        desired_end_states_2 = torch.tensor([-7.0106, 2, 2])
        desired_end_states_3 = torch.tensor([-8.5362, 3, 0])
        np.testing.assert_array_equal(
            all_end_states[0, :].numpy(), desired_end_states_1.numpy()
        )
        np.testing.assert_array_equal(
            all_end_states[1, :].numpy(), desired_end_states_2.numpy()
        )
        np.testing.assert_array_equal(
            all_end_states[2, :].numpy(), desired_end_states_3.numpy()
        )

    def test_get_all_end_states_with_same_scores(self):
        """
        test the function of keeping nBest end_states
        When two hypotheses have same scores, keep the first one.
        """
        beam_tokens = torch.tensor([[2, 2, 2, 2], [3, 4, 5, 6]])
        beam_scores = torch.tensor(
            [[0.0000, 0.0000, 0.0000, 0.0000], [-4.4000, -4.4000, -4.2000, -4.3000]]
        )
        beam_prev_indices = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
        beam_decode = BeamDecode(
            eos_token_id=2, length_penalty=0.0, nbest=3, beam_size=4, stop_at_eos=True
        )

        all_end_states = beam_decode._get_all_end_states(
            beam_tokens, beam_scores, beam_prev_indices, num_steps=1
        )

        desired_end_states_1 = torch.tensor([-4.2000, 1, 2])
        desired_end_states_2 = torch.tensor([-4.3000, 1, 3])
        # keep the first hypo [-4.4000, 1, 0] rather than [-4.4000, 1, 1]
        desired_end_states_3 = torch.tensor([-4.4000, 1, 0])
        np.testing.assert_array_equal(
            all_end_states[0, :].numpy(), desired_end_states_1.numpy()
        )
        np.testing.assert_array_equal(
            all_end_states[1, :].numpy(), desired_end_states_2.numpy()
        )
        np.testing.assert_array_equal(
            all_end_states[2, :].numpy(), desired_end_states_3.numpy()
        )
