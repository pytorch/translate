#!/usr/bin/env python3

import shutil
import tempfile
import unittest
from os import path
from unittest.mock import Mock, patch

from pytorch_translate.research.unsupervised_morphology import (
    unsupervised_bilingual_morphology,
)


def get_two_tmp_files():
    src_txt_content = [
        "123 124 234 345",
        "112 122 123 345",
        "123456789",
        "123456 456789",
    ]
    dst_txt_content = [
        "123 124 234 345",
        "112 122 123 345",
        "123456789",
        "123456 456789",
    ]
    content1, content2 = "\n".join(src_txt_content), "\n".join(dst_txt_content)
    tmp_dir = tempfile.mkdtemp()
    file1, file2 = path.join(tmp_dir, "test1.txt"), path.join(tmp_dir, "test2.txt")
    with open(file1, "w") as f1:
        f1.write(content1)
    with open(file2, "w") as f2:
        f2.write(content2)

    return tmp_dir, file1, file2


class TestUnsupervisedBilingualMorphology(unittest.TestCase):
    def test_morph_init(self):
        morph_hmm_model = (
            unsupervised_bilingual_morphology.BilingualMorphologyHMMParams()
        )
        tmp_dir, f1, f2 = get_two_tmp_files()
        morph_hmm_model.init_params_from_data(
            src_path=f1, dst_path=f2, num_ibm_iters=3, num_candidate_per_word=2
        )
        assert len(morph_hmm_model.alignment_probs) == 7
        assert len(morph_hmm_model.alignment_probs["123456"]) == 2
        assert len(morph_hmm_model.alignment_probs["123456789"]) == 1

        morph_hmm_model.init_params_from_data(
            src_path=f1, dst_path=f2, num_ibm_iters=3, num_candidate_per_word=10
        )
        assert len(morph_hmm_model.alignment_probs) == 9
        assert len(morph_hmm_model.alignment_probs["123"]) == 6
        shutil.rmtree(tmp_dir)
