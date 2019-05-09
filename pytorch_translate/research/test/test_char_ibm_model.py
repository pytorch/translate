#!/usr/bin/env python3

import shutil
import unittest
from collections import Counter
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
        assert len(substrs) == 34
        substr_counter = Counter(substrs)
        assert len(substr_counter) == 24
        assert substr_counter[char_ibm_model.str2int(char_ibm_model.eow_symbol)] == 1
        assert (
            substr_counter[char_ibm_model.str2int("5" + char_ibm_model.eow_symbol)] == 1
        )
        assert substr_counter[char_ibm_model.str2int("123")] == 2
        assert "12345" not in char_ibm_model._str2int

    def test_get_subwords_counts_for_line(self):
        char_ibm_model = CharIBMModel1(max_subword_len=4)

        substrs = char_ibm_model.get_subword_counts_for_line("123412345 12345")
        assert len(substrs) == 24
        assert substrs[char_ibm_model.str2int(char_ibm_model.eow_symbol)] == 2
        assert substrs[char_ibm_model.str2int("5" + char_ibm_model.eow_symbol)] == 2
        assert substrs[char_ibm_model.str2int("123")] == 3
        assert "12341" not in char_ibm_model._str2int

    def test_morph_init(self):
        tmp_dir, f1, f2 = morph_utils.get_two_different_tmp_files()

        ibm_model = CharIBMModel1()
        ibm_model.initialize_translation_probs(f1, f2)
        assert (
            ibm_model.translation_prob[ibm_model.str2int("5")][
                ibm_model.str2int("d" + ibm_model.eow_symbol)
            ]
            > 0
        )
        assert len(ibm_model.translation_prob) == 80

        ibm_model = Word2CharIBMModel1(max_subword_len=4)
        ibm_model.initialize_translation_probs(f1, f2)
        assert (
            ibm_model.str2int("abcdefghi")
            not in ibm_model.translation_prob[ibm_model.str2int("123456789")]
        )
        assert (
            ibm_model.str2int("cdef")
            in ibm_model.translation_prob[ibm_model.str2int("123456789")]
        )
        assert (
            ibm_model.str2int("cde")
            in ibm_model.translation_prob[ibm_model.str2int("123456789")]
        )
        assert len(ibm_model.translation_prob[ibm_model.str2int("123456789")]) == 34
        assert len(ibm_model.translation_prob) == 10

        shutil.rmtree(tmp_dir)
