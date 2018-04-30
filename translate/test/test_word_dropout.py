#!/usr/bin/env python3

import unittest

from pytorch_translate import word_dropout
from pytorch_translate.test import utils as test_utils


class TestWordDropout(unittest.TestCase):
    def test_apply_probabilistic_unking(self):
        word_dropout_params = {
            'word_dropout_freq_threshold': 3,
            'word_dropout_smoothing_alpha': 1,
        }
        test_args = test_utils.ModelParamsDict(
            cell_type='rnn',
        )
        test_args.word_dropout_params = word_dropout_params
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        word_dropout_module = \
            word_dropout.WordDropout(src_dict, word_dropout_params)
        word_dropout_module.apply_probabilistic_unking(3)
