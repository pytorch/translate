#!/usr/bin/env python3

import shutil
import tempfile
import unittest
from os import path

from pytorch_translate.research.unsupervised_morphology.unsupervised_bilingual_morphology import (
    BilingualMorphologyHMMParams,
)


src_txt_content = ["123 124 234 345", "112 122 123 345", "123456789", "123456 456789"]
dst_txt_content = ["123 124 234 345", "112 122 123 345", "123456789", "123456 456789"]


def get_two_tmp_files(content1, content2):
    tmp_dir = tempfile.mkdtemp()
    file1, file2 = path.join(tmp_dir, "test1.txt"), path.join(tmp_dir, "test2.txt")
    f1 = open(file1, "w")
    f1.write(content1)
    f1.close()

    f2 = open(file2, "w")
    f2.write(content2)
    f2.close()

    return tmp_dir, file1, file2


class TestUnsupervisedMorphology(unittest.TestCase):
    def test_morph_init(self):
        morph_hmm_model = BilingualMorphologyHMMParams()

        tmp_dir, f1, f2 = get_two_tmp_files(
            "\n".join(src_txt_content), "\n".join(dst_txt_content)
        )
        morph_hmm_model.init_params_from_data(f1, f2)
        print(len(morph_hmm_model.morph_emit_probs))
        assert len(morph_hmm_model.morph_emit_probs) == 200
        assert round(morph_hmm_model.morph_emit_probs["1234"], 3) == round(
            0.0062799043062200955, 3
        )
        assert round(morph_hmm_model.translation_probs["1234"]["1234"], 3) == 1.0 / 200
        shutil.rmtree(tmp_dir)

    def test_zero_out_params(self):
        morph_hmm_model = BilingualMorphologyHMMParams()

        tmp_dir, f1, f2 = get_two_tmp_files(
            "\n".join(src_txt_content), "\n".join(dst_txt_content)
        )
        morph_hmm_model.init_params_from_data(f1, f2)
        for morph in morph_hmm_model.translation_probs.keys():
            assert morph_hmm_model.morph_emit_probs[morph] > 0
            for target_morph in morph_hmm_model.translation_probs.keys():
                assert morph_hmm_model.translation_probs[morph][target_morph] > 0

        morph_hmm_model.zero_out_parmas()
        for morph in morph_hmm_model.morph_emit_probs.keys():
            assert morph_hmm_model.morph_emit_probs[morph] == 0
            for target_morph in morph_hmm_model.translation_probs.keys():
                assert morph_hmm_model.translation_probs[morph][target_morph] == 0
        shutil.rmtree(tmp_dir)

    def test_save_load(self):
        morph_hmm_model = BilingualMorphologyHMMParams()

        tmp_dir, f1, f2 = get_two_tmp_files(
            "\n".join(src_txt_content), "\n".join(dst_txt_content)
        )
        morph_hmm_model.init_params_from_data(f1, f2)
        morph_hmm_model.save(path.join(tmp_dir, "test.pickle"))
        loaded_params = BilingualMorphologyHMMParams.load(
            path.join(tmp_dir, "test.pickle")
        )

        assert morph_hmm_model.morph_emit_probs == loaded_params.morph_emit_probs
        assert morph_hmm_model.smoothing_const == loaded_params.smoothing_const
        assert morph_hmm_model.SMALL_CONST == loaded_params.SMALL_CONST
        assert morph_hmm_model.len_cost_pow == loaded_params.len_cost_pow
        assert morph_hmm_model.max_morph_len == loaded_params.max_morph_len
        assert morph_hmm_model.translation_probs == loaded_params.translation_probs
        shutil.rmtree(tmp_dir)
