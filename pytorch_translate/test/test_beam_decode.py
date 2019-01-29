#!/usr/bin/env python3

import unittest
from typing import Any, List

import numpy as np
import torch
from pytorch_translate import char_source_model  # noqa
from pytorch_translate import rnn  # noqa
from pytorch_translate import beam_decode, generate
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestBeamDecode(unittest.TestCase):
    def test_basic_generate(self):
        test_args = test_utils.ModelParamsDict()
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        model = task.build_model(test_args)
        translator = beam_decode.SequenceGenerator([model], task.target_dictionary)
        src_tokens = torch.LongTensor([[0, 0, 0], [0, 0, 0]])
        src_lengths = torch.LongTensor([3, 3])
        encoder_input = {"src_tokens": src_tokens, "src_lengths": src_lengths}
        translator.generate(encoder_input, maxlen=7)

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
        translator = beam_decode.SequenceGenerator(
            [model], task.target_dictionary, use_char_source=True
        )
        src_tokens = torch.LongTensor([[0, 0, 0], [0, 0, 0]])
        src_lengths = torch.LongTensor([3, 3])
        char_inds = torch.LongTensor(np.zeros((2, 3, 5)))
        word_lengths = torch.LongTensor([[5, 5, 5], [5, 5, 5]])
        encoder_input = {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "char_inds": char_inds,
            "word_lengths": word_lengths,
        }
        translator.generate(encoder_input, maxlen=7)

    def test_gather_probs_with_vr(self):
        """ Tests gather_probs when there is vocab reduction """
        all_translation_tokens: List[Any] = [
            torch.LongTensor([3, 7, 8, 9]),
            torch.LongTensor([0, 3, 5]),
        ]
        all_probs: List[Any] = [
            torch.FloatTensor([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]),
            torch.FloatTensor([[0.4, 0.5, 0.1], [0.4, 0.5, 0.1]]),
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

    def test_smoothed_sentence_bleu(self):
        """
        Testing calculation of smoothed_sentence_bleu() function.
        Inputs:
            target_tokens: [11, 12, 13, 14, 15]
            hypo_tokens: [11, 12, 14, 15]
            actual precision:
                unigram: 4/4 = 1
                bigram:  2/3 = 0.667
                trigram: 0/2 = 0
                4-gram:  0/1 = 0
            smoothed precision:
                unigram: 4/4    = 1
                bigram:  2/3    = 0.667
                trigram: 0.5/2  = 0.25
                4-gram:  0.25/1 = 0.25
            smoothed geom. mean: (1 * 2/3 * 1/4 * 1/4) ^ (1/4) = 0.4518
            brevity penalty: e ^ (1 - 5/4) = 0.7788
        Desired Output:
            0.4518 * 0.7788 = 0.35186
        """
        test_args = test_utils.ModelParamsDict()
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        target_tokens = torch.IntTensor([11, 12, 13, 14, 15])
        hypo_tokens = torch.IntTensor([11, 12, 14, 15])
        smoothed_bleu = generate.smoothed_sentence_bleu(
            task, target_tokens, hypo_tokens
        )
        np.testing.assert_almost_equal(smoothed_bleu, 0.35186, decimal=5)
