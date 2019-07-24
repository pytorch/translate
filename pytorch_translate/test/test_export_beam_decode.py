#!/usr/bin/env python3

import unittest

import numpy as np
import torch
from pytorch_translate import char_source_model  # noqa
from pytorch_translate import rnn  # noqa
from pytorch_translate.beam_decode import BeamDecode, SequenceGenerator
from pytorch_translate.ensemble_export import BeamSearch
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


MAX_VOCAB_SIZE = 7


def prepare_test_input(src_length=10, num_step=20, beam_size=5):
    beam_tokens = torch.randint(MAX_VOCAB_SIZE, (num_step + 1, beam_size))
    beam_scores = torch.rand((num_step + 1, beam_size))
    token_weights = torch.rand((num_step + 1, beam_size, src_length))
    beam_prev_indices = torch.randint(beam_size, (num_step + 1, beam_size))
    num_step = num_step
    return (beam_tokens, beam_scores, token_weights, beam_prev_indices, num_step)


class TestExportBeamDecode(unittest.TestCase):
    def test_beamsize_one_decode(self):
        beam_decode = BeamDecode(
            eos_token_id=2, length_penalty=0.25, nbest=5, beam_size=1, stop_at_eos=True
        )
        one_beamsize_input = prepare_test_input(beam_size=1)

        beam_tokens, beam_scores, token_weights, beam_prev_indices, num_step = (
            one_beamsize_input
        )
        output = beam_decode(
            beam_tokens, beam_scores, token_weights, beam_prev_indices, num_step
        )

        top_hypothesis_tokens = output[0][0]

        # 0 index row is ignored
        beam_search_token_output = beam_tokens.reshape(-1)[1:].tolist()
        if beam_decode.eos_token_id in beam_search_token_output:
            index_of_eos = beam_search_token_output.index(beam_decode.eos_token_id)
            beam_search_token_output = beam_search_token_output[: index_of_eos + 1]

        np.testing.assert_array_equal(top_hypothesis_tokens, beam_search_token_output)

    def test_attention_weights(self):
        beam_decode = BeamDecode(
            eos_token_id=2, length_penalty=0.25, nbest=5, beam_size=4, stop_at_eos=True
        )
        one_beamsize_input = prepare_test_input(beam_size=4)

        beam_tokens, beam_scores, token_weights, beam_prev_indices, num_step = (
            one_beamsize_input
        )
        output = beam_decode(
            beam_tokens, beam_scores, token_weights, beam_prev_indices, num_step
        )

        all_end_states = beam_decode._get_all_end_states(
            beam_tokens, beam_scores, beam_prev_indices, num_step
        )
        for state_idx, end_state in enumerate(all_end_states):
            beam_indices = beam_decode._get_output_steps_to_beam_indices(
                end_state, beam_prev_indices
            )
            weights_from_output = output[state_idx][3].numpy()
            weights_from_input = []
            for pos, beam_index in enumerate(beam_indices):
                if pos == 0:
                    continue
                weights_from_input.append(token_weights[pos][beam_index])
            weights_from_input = torch.stack(weights_from_input, dim=1).numpy()

            np.testing.assert_array_equal(weights_from_output, weights_from_input)

    def test_known_inputs_outputs(self):
        ALL_TESTS = [End2EndTest()]

        for test in ALL_TESTS:
            beam_decode = BeamDecode(
                eos_token_id=test.eos_token_id,
                length_penalty=test.length_penalty,
                nbest=test.nbest,
                beam_size=test.beam_size,
                stop_at_eos=test.stop_at_eos,
            )
            output = beam_decode(*test.prepare_input())
            test.test_output(output)

    def test_basic_generate(self):
        """
        A basic test that the output given by SequenceGenerator class is the same
        """
        # Setup parameters required for SequenceGenerator and BeamSeach/BeamDecode
        TEST_ARGS = test_utils.ModelParamsDict(arch="rnn")
        TEST_ARGS.sequence_lstm = True
        BEAM_SIZE = 1
        WORD_REWARD = 0
        UNK_REWARD = 0
        LENGTH_PENALTY = 0

        PLACEHOLDER_SEQ_LENGTH = 5
        NBEST = 2
        MAX_SEQ_LEN = 7

        src_tokens = torch.LongTensor([[0, 0, 0]])
        src_lengths = torch.LongTensor([3])

        # Generate values using SequenceGenerator
        _, src_dict, tgt_dict = test_utils.prepare_inputs(TEST_ARGS)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        model = task.build_model(TEST_ARGS)
        translator = SequenceGenerator(
            [model],
            task.target_dictionary,
            beam_size=BEAM_SIZE,
            word_reward=WORD_REWARD,
            unk_reward=UNK_REWARD,
        )

        encoder_input = {"src_tokens": src_tokens, "src_lengths": src_lengths}
        top_seq_gen_hypothesis = translator.generate(encoder_input, maxlen=MAX_SEQ_LEN)[
            0
        ]

        # Generate output using BeamSearch/BeamDecode
        placeholder_src_tokens = torch.LongTensor(
            np.ones((PLACEHOLDER_SEQ_LENGTH, 1), dtype="int64")
        )
        placeholder_src_lengths = torch.IntTensor(
            np.array([PLACEHOLDER_SEQ_LENGTH], dtype="int32")
        )

        beam_search = BeamSearch(
            [model],
            tgt_dict,
            placeholder_src_tokens,
            placeholder_src_lengths,
            beam_size=BEAM_SIZE,
            word_reward=WORD_REWARD,
            unk_reward=UNK_REWARD,
            quantize=False,
        )
        beam_decode = BeamDecode(
            eos_token_id=tgt_dict.eos(),
            length_penalty=LENGTH_PENALTY,
            nbest=NBEST,
            beam_size=BEAM_SIZE,
            stop_at_eos=True,
        )

        # Few more placeholder inputs for BeamSearch
        prev_token = torch.LongTensor([tgt_dict.eos()])
        prev_scores = torch.FloatTensor([0.0])
        attn_weights = torch.zeros(src_lengths[0].item())
        prev_hypos_indices = torch.zeros(1, dtype=torch.int64)
        num_steps = torch.LongTensor([MAX_SEQ_LEN])

        all_tokens, all_scores, all_weights, all_prev_indices = beam_search(
            src_tokens.transpose(0, 1),
            src_lengths,
            prev_token,
            prev_scores,
            attn_weights,
            prev_hypos_indices,
            num_steps,
        )
        beam_decode_output = beam_decode(
            all_tokens, all_scores, all_weights, all_prev_indices, num_steps[0]
        )

        for hyp_index in range(
            min(len(beam_decode_output), len(top_seq_gen_hypothesis))
        ):
            top_beam_decode_hypothesis = beam_decode_output[hyp_index]

            # Compare two outputs
            # We always look only from 0 to MAX_SEQ_LEN, because sequence generator
            # adds an EOS at the end after MAX_SEQ_LEN
            ## Compare two hypothesis
            np.testing.assert_array_equal(
                top_seq_gen_hypothesis[hyp_index]["tokens"].tolist()[0:MAX_SEQ_LEN],
                top_beam_decode_hypothesis[0].tolist()[0:MAX_SEQ_LEN],
            )
            ## Compare token level scores
            np.testing.assert_array_almost_equal(
                top_seq_gen_hypothesis[hyp_index]["positional_scores"].tolist()[
                    0:MAX_SEQ_LEN
                ],
                top_beam_decode_hypothesis[2][0:MAX_SEQ_LEN],
                decimal=1,
            )

            ## Compare attention weights
            np.testing.assert_array_almost_equal(
                top_seq_gen_hypothesis[hyp_index]["attention"].numpy()[
                    :, 0:MAX_SEQ_LEN
                ],
                top_beam_decode_hypothesis[3].numpy()[:, 0:MAX_SEQ_LEN],
                decimal=1,
            )
            ## Not testing the hypothesis score as sequence generator is adding EOS
            ## at the end, it changes the final score


class BeamDecodeTestBase(object):
    def __init__(self):
        self.eos_token_id = 2
        self.length_penalty = 0.25
        self.nbest = 2
        self.beam_size = 3
        self.stop_at_eos = True

    def prepare_input(self):
        pass

    def test_output(self):
        pass


class End2EndTest(BeamDecodeTestBase):
    def prepare_input(self):
        self.src_length = 3
        self.num_steps = 5

        beam_tokens = torch.tensor(
            [[0, 0, 0], [0, 4, 1], [6, 2, 3], [1, 1, 0], [2, 1, 5], [3, 5, 5]]
        )
        beam_scores = torch.tensor(
            [
                [0.2322, 0.3070, 0.3755],
                [0.8971, 0.0688, 0.9484],
                [0.2515, 0.2836, 0.2849],
                [0.1710, 0.6512, 0.6899],
                [0.1411, 0.1590, 0.0182],
                [0.262, 0.193, 0.015],
            ]
        )
        token_weights = torch.tensor(
            [
                [
                    [0.0378, 0.1570, 0.4723],
                    [0.6100, 0.0196, 0.9939],
                    [0.2908, 0.1187, 0.3387],
                ],
                [
                    [0.1265, 0.7551, 0.1549],
                    [0.8359, 0.3574, 0.2500],
                    [0.4979, 0.0281, 0.8424],
                ],
                [
                    [0.9921, 0.0731, 0.5912],
                    [0.0423, 0.4134, 0.9655],
                    [0.9652, 0.0211, 0.7168],
                ],
                [
                    [0.8314, 0.9484, 0.4877],
                    [0.8744, 0.2184, 0.6247],
                    [0.4664, 0.3436, 0.1755],
                ],
                [
                    [0.2532, 0.3367, 0.3911],
                    [0.2439, 0.8911, 0.7692],
                    [0.9397, 0.1810, 0.3092],
                ],
                [
                    [0.3943, 0.1062, 0.4711],
                    [0.0387, 0.0143, 0.0836],
                    [0.7161, 0.6038, 0.4064],
                ],
            ]
        )
        beam_prev_indices = torch.tensor(
            [[1, 1, 2], [2, 0, 0], [1, 0, 2], [2, 2, 0], [2, 0, 0], [0, 2, 0]]
        )
        return (
            beam_tokens,
            beam_scores,
            token_weights,
            beam_prev_indices,
            self.num_steps,
        )

    def test_output(self, output):
        np.testing.assert_array_equal(output[0][0].numpy(), np.array([0, 2]))
        np.testing.assert_array_equal(output[1][0].numpy(), np.array([1, 3, 1, 5, 5]))
        # Third best candidate is [4, 6, 0, 2]
