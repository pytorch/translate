#!/usr/bin/env python3

import shutil
import tempfile
import unittest
from os import path

from pytorch_translate.research.unsupervised_morphology.ibm_model1 import IBMModel1


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


class TestIBMModel1(unittest.TestCase):
    def test_morph_init(self):
        ibm_model = IBMModel1()

        tmp_dir, f1, f2 = get_two_tmp_files()
        ibm_model.initialize_translation_probs(f1, f2)
        assert len(ibm_model.translation_prob) == 10
        assert len(ibm_model.translation_prob[ibm_model.null_str]) == 9
        assert len(ibm_model.translation_prob["345"]) == 6
        assert ibm_model.translation_prob["122"]["123"] == 1.0 / 4
        shutil.rmtree(tmp_dir)
