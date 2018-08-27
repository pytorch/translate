#!/usr/bin/env python3

import unittest
from typing import Any, List

import numpy as np
import torch
from pytorch_translate import char_source_model  # noqa (must be after rnn)
from pytorch_translate import rnn  # noqa
from pytorch_translate import beam_decode, tasks
from pytorch_translate.test import utils as test_utils


class TestBeamDecode(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_basic_generate(self):
        test_args = test_utils.ModelParamsDict()
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        model = task.build_model(test_args)
        translator = beam_decode.SequenceGenerator([model], task.target_dictionary)
        src_tokens = torch.LongTensor([[0, 0, 0], [0, 0, 0]])
        src_lengths = torch.LongTensor([3, 3])
        encoder_input = (src_tokens, src_lengths)
        translator.generate(encoder_input, maxlen=7)

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_char_rnn_generate(self):
        test_args = test_utils.ModelParamsDict(sequence_lstm=True)
        test_args.arch = "char_source"
        test_args.char_source_dict_size = 126
        test_args.char_embed_dim = 8
        test_args.char_rnn_units = 12
        test_args.char_rnn_layers = 2

        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        model = task.build_model(test_args)
        translator = beam_decode.SequenceGenerator([model], task.target_dictionary)
        src_tokens = torch.LongTensor([[0, 0, 0], [0, 0, 0]])
        src_lengths = torch.LongTensor([3, 3])
        char_inds = torch.LongTensor(np.zeros((2, 3, 5)))
        word_lengths = torch.LongTensor([[5, 5, 5], [5, 5, 5]])
        encoder_input = (src_tokens, src_lengths, char_inds, word_lengths)
        translator.generate(encoder_input, maxlen=7)

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_gather_probs_with_vr(self):
        """ Tests gather_probs when there is vocab reduction """
        all_translation_tokens: List[Any] = [
            torch.LongTensor([3, 7, 8, 9]),
            torch.LongTensor([0, 3, 5]),
        ]
        all_probs: List[Any] = [
            torch.FloatTensor(
                [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
            ).cuda(),
            torch.FloatTensor([[0.4, 0.5, 0.1], [0.4, 0.5, 0.1]]).cuda(),
        ]
        avg_probs, possible_translation_tokens = beam_decode.SequenceGenerator.gather_probs(
            all_translation_tokens=all_translation_tokens, all_probs=all_probs
        )
        avg_probs = avg_probs.detach().cpu().numpy()
        possible_translation_tokens = possible_translation_tokens.detach().cpu().numpy()

        avg_probs_ref = sorted([0.4, 0.75, 0.1, 0.25, 0.25, 0.25])
        possible_translation_tokens_ref = sorted([0, 3, 5, 7, 8, 9])

        np.testing.assert_allclose(
            actual=np.sort(avg_probs[0]), desired=np.array(avg_probs_ref), atol=1e-5
        )
        np.testing.assert_allclose(
            actual=np.sort(possible_translation_tokens),
            desired=np.array(possible_translation_tokens_ref),
        )
        np.testing.assert_allclose(
            actual=np.sort(possible_translation_tokens),
            desired=np.array(possible_translation_tokens_ref),
        )

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_gather_probs_without_vr(self):
        """ Tests gather_probs when there is no vocab reduction """
        all_probs: List[Any] = [
            torch.FloatTensor([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]),
            torch.FloatTensor([[0.4, 0.2, 0.1, 0.3], [0.4, 0.2, 0.1, 0.3]]),
        ]
        all_translation_tokens: List[Any] = [None, None]
        avg_probs, possible_translation_tokens = beam_decode.SequenceGenerator.gather_probs(
            all_translation_tokens=all_translation_tokens, all_probs=all_probs
        )

        assert possible_translation_tokens is None
        avg_probs_ref = [0.65, 0.45, 0.35, 0.55]
        np.testing.assert_allclose(
            actual=avg_probs[0], desired=np.array(avg_probs_ref), atol=1e-5
        )
        np.testing.assert_allclose(
            actual=avg_probs[1], desired=np.array(avg_probs_ref), atol=1e-5
        )
