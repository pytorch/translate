#!/usr/bin/env python3

import shutil
import unittest
from multiprocessing import Pool

from pytorch_translate.research.test import morphology_test_utils as morph_utils
from pytorch_translate.research.unsupervised_morphology.char_ibm_model1 import (
    CharIBMModel1,
    Word2CharIBMModel1,
)


class TestCharIBMModel1(unittest.TestCase):
    def test_get_possible_subwords(self):
        char_ibm_model = CharIBMModel1(max_subword_len=4)

        substrs = char_ibm_model.get_possible_subwords("123412345")
        assert len(substrs) == 24
        assert substrs[char_ibm_model.eow_symbol] == 1
        assert substrs["5" + char_ibm_model.eow_symbol] == 1
        assert substrs["123"] == 2
        assert "12345" not in substrs

    def test_get_subwords_counts_for_line(self):
        char_ibm_model = CharIBMModel1(max_subword_len=4)

        substrs = char_ibm_model.get_subword_counts_for_line("123412345 12345")
        assert len(substrs) == 24
        assert substrs[char_ibm_model.eow_symbol] == 2
        assert substrs["5" + char_ibm_model.eow_symbol] == 2
        assert substrs["123"] == 3
        assert "12345" not in substrs

    def test_morph_init(self):
        tmp_dir, f1, f2 = morph_utils.get_two_different_tmp_files()

        ibm_model = CharIBMModel1()
        ibm_model.initialize_translation_probs(f1, f2)
        assert ibm_model.translation_prob["5"]["d" + ibm_model.eow_symbol] > 0
        assert len(ibm_model.translation_prob) == 80
        assert len(ibm_model.training_data) == 4

        ibm_model = Word2CharIBMModel1(max_subword_len=4)
        ibm_model.initialize_translation_probs(f1, f2)
        assert "abcdefghi" not in ibm_model.translation_prob["123456789"]
        assert "cdef" in ibm_model.translation_prob["123456789"]
        assert "cde" in ibm_model.translation_prob["123456789"]
        assert len(ibm_model.translation_prob["123456789"]) == 34
        assert len(ibm_model.translation_prob) == 9
        assert len(ibm_model.training_data) == 4

        shutil.rmtree(tmp_dir)
