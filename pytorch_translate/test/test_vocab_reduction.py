#!/usr/bin/env python3

import unittest

import numpy as np
from pytorch_translate import vocab_reduction
from pytorch_translate.test import utils as test_utils


class TestVocabReduction(unittest.TestCase):
    def test_get_translation_candidates(self):
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        src_dict, dst_dict = test_utils.create_vocab_dictionaries()
        translation_candidates = vocab_reduction.get_translation_candidates(
            src_dict=src_dict,
            dst_dict=dst_dict,
            lexical_dictionaries=lexical_dictionaries,
            num_top_words=10,
            max_translation_candidates_per_word=1,
        )

        translation_candidates_ref = test_utils.create_vocab_reduction_expected_array(
            src_dict
        )
        assert translation_candidates.size != 0
        np.testing.assert_array_equal(
            translation_candidates, translation_candidates_ref
        )
