#!/usr/bin/env python3

import shutil
import tempfile
import unittest
from collections import Counter, defaultdict
from os import path

from pytorch_translate.research.test import morphology_test_utils as morph_utils
from pytorch_translate.research.unsupervised_morphology.ibm_model1 import IBMModel1


class TestIBMModel1(unittest.TestCase):
    def test_str2int(self):
        ibm_model = IBMModel1()
        # Calling multiple times to make sure we get the same value.
        assert ibm_model.str2int("hello") == 1
        assert ibm_model.str2int("bye") == 2
        assert ibm_model.str2int("hello") == 1
        assert ibm_model.str2int("bye") == 2
        assert len(ibm_model._str2int) == 3
        assert len(ibm_model._int2str) == 3
        assert ibm_model._int2str == [ibm_model.null_str, "hello", "bye"]
        assert ibm_model.int2str(2) == "bye"

    def test_morph_init(self):
        ibm_model = IBMModel1()

        tmp_dir, f1, f2 = morph_utils.get_two_same_tmp_files()
        ibm_model.initialize_translation_probs(f1, f2)
        assert len(ibm_model.translation_prob) == 10
        assert (
            len(ibm_model.translation_prob[ibm_model.str2int(ibm_model.null_str)]) == 9
        )
        assert len(ibm_model.translation_prob[ibm_model.str2int("345")]) == 6
        assert (
            ibm_model.translation_prob[ibm_model.str2int("122")][
                ibm_model.str2int("123")
            ]
            == 1.0 / 4
        )
        shutil.rmtree(tmp_dir)

    def test_expectation_for_one_sentence(self):
        ibm_model = IBMModel1()

        tmp_dir, f1, f2 = morph_utils.get_two_same_tmp_files()
        ibm_model.initialize_translation_probs(f1, f2)
        translation_counts = defaultdict(lambda: defaultdict(float))

        ibm_model.expectation_for_one_sentence(
            Counter(
                ibm_model.str2int(w)
                for w in ["123", "124", "234", "345", ibm_model.null_str]
            ),
            Counter(ibm_model.str2int(w) for w in ["123", "124", "234", "345"]),
            translation_counts,
        )
        assert (
            round(
                translation_counts[ibm_model.str2int("123")][ibm_model.str2int("345")],
                3,
            )
            == 0.176
        )
        shutil.rmtree(tmp_dir)

    def test_ibm_train(self):
        ibm_model = IBMModel1()

        tmp_dir, f1, f2 = morph_utils.get_two_same_tmp_files()
        ibm_model.learn_ibm_parameters(src_path=f1, dst_path=f2, num_iters=3)

        assert (
            ibm_model.translation_prob[ibm_model.str2int("456789")][
                ibm_model.str2int("345")
            ]
            == 0
        )
        assert (
            ibm_model.translation_prob[ibm_model.str2int("456789")][
                ibm_model.str2int("456789")
            ]
            == 0.5
        )
        shutil.rmtree(tmp_dir)
