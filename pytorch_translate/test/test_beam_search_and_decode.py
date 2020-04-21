#!/usr/bin/env python3

import unittest

import numpy as np
import torch
from pytorch_translate import rnn  # noqa
from pytorch_translate.beam_decode import BeamDecode
from pytorch_translate.beam_search_and_decode_v2 import BeamDecodeWithEOS


class TestBeamSearchAndDecode(unittest.TestCase):
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

    def test_beam_decode_with_eos(self):
        beam_tokens = torch.tensor(
            [
                [2, 2, 2, 2],
                [3, 4, 5, 2],
                [2, 6, 6, 2],
                [7, 2, 1, 8],
                [9, 9, 8, 1],
                [2, 2, 2, 2],
            ]
        )
        beam_prev_indices = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 1],
                [1, 1, 2, 2],
                [0, 2, 0, 2],
                [0, 1, 2, 3],
            ]
        )
        beam_scores = torch.tensor(
            [
                [0.00, 0.00, 0.00, 0.00],
                [-3.1, -3.2, -3.3, -4.2],
                [-7.4, -6.3, -6.7, -7.5],
                [-10.6, -11.7, -10.8, -10.7],
                [-14.2, -15.5, -14.1, -14.9],
                [-18.3, -18.1, -18.4, -18.8],
            ]
        )

        # test end_states generation

        # since we only consider eos when it's among the top beam_size indices,
        # [-4.2, 1, 3], [-7.5, 2, 3] shouldn't be generated.
        desired_end_states_1 = torch.tensor([-7.4, 2, 0])
        desired_end_states_2 = torch.tensor([-11.7, 3, 1])
        desired_end_states_3 = torch.tensor([-18.1, 5, 1])
        desired_end_states_4 = torch.tensor([-18.3, 5, 0])

        beam_decode_with_eos = BeamDecodeWithEOS(
            eos_token_id=2, length_penalty=0.0, nbest=10, beam_size=2, stop_at_eos=True
        )
        all_end_states = beam_decode_with_eos._get_all_end_states(
            beam_tokens, beam_scores, beam_prev_indices, num_steps=4
        )

        np.testing.assert_array_equal(
            all_end_states[0, :].numpy(), desired_end_states_1.numpy()
        )
        np.testing.assert_array_equal(
            all_end_states[1, :].numpy(), desired_end_states_2.numpy()
        )
        np.testing.assert_array_equal(
            all_end_states[2, :].numpy(), desired_end_states_3.numpy()
        )
        np.testing.assert_array_equal(
            all_end_states[3, :].numpy(), desired_end_states_4.numpy()
        )

        # test hypotheses generation
        token_weights = torch.rand((6, 4, 5))

        all_hypotheses = beam_decode_with_eos(
            beam_tokens, beam_scores, token_weights, beam_prev_indices, num_steps=4
        )

        np.testing.assert_array_equal([3, 2], all_hypotheses[0][0])
        np.testing.assert_array_equal([4, 6, 2], all_hypotheses[1][0])
        np.testing.assert_array_equal([3, 6, 1, 9, 2], all_hypotheses[2][0])
        np.testing.assert_array_equal([4, 6, 7, 9, 2], all_hypotheses[3][0])
