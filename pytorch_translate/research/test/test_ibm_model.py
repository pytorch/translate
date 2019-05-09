#!/usr/bin/env python3

import shutil
import tempfile
import unittest
from collections import Counter, defaultdict
from os import path

from pytorch_translate.research.test import morphology_test_utils as morph_utils
from pytorch_translate.research.unsupervised_morphology.ibm_model1 import IBMModel1


class TestIBMModel1(unittest.TestCase):
    def test_morph_init(self):
        ibm_model = IBMModel1()

        tmp_dir, f1, f2 = morph_utils.get_two_same_tmp_files()
        ibm_model.initialize_translation_probs(f1, f2)
        assert len(ibm_model.translation_prob) == 10
        assert len(ibm_model.translation_prob[ibm_model.null_str]) == 9
        assert len(ibm_model.translation_prob["345"]) == 6
        assert ibm_model.translation_prob["122"]["123"] == 1.0 / 4
        shutil.rmtree(tmp_dir)

    def test_expectation_for_one_sentence(self):
        ibm_model = IBMModel1()

        tmp_dir, f1, f2 = morph_utils.get_two_same_tmp_files()
        ibm_model.initialize_translation_probs(f1, f2)
        translation_counts = defaultdict(lambda: defaultdict(float))

        ibm_model.expectation_for_one_sentence(
            Counter(["123", "124", "234", "345", ibm_model.null_str]),
            Counter(["123", "124", "234", "345"]),
            translation_counts,
        )
        assert round(translation_counts["123"]["345"], 3) == 0.176
        shutil.rmtree(tmp_dir)

    def test_ibm_train(self):
        ibm_model = IBMModel1()

        tmp_dir, f1, f2 = morph_utils.get_two_same_tmp_files()
        ibm_model.learn_ibm_parameters(src_path=f1, dst_path=f2, num_iters=3)

        assert ibm_model.translation_prob["456789"]["345"] == 0
        assert ibm_model.translation_prob["456789"]["456789"] == 0.5
        shutil.rmtree(tmp_dir)
