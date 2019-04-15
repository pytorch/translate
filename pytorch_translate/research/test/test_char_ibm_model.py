#!/usr/bin/env python3

import unittest

from pytorch_translate.research.unsupervised_morphology.char_ibm_model1 import (
    CharIBMModel1,
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
