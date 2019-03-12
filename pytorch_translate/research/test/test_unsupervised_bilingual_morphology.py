#!/usr/bin/env python3

import math
import shutil
import tempfile
import unittest
from os import path

from pytorch_translate.research.unsupervised_morphology.unsupervised_bilingual_morphology import (
    BilingualMorphologyHMMParams,
    BilingualMorphologySegmentor,
    UnsupervisedBilingualMorphology,
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
        morph_hmm_model = BilingualMorphologyHMMParams()

        tmp_dir, f1, f2 = get_two_tmp_files()
        morph_hmm_model.init_params_from_data(f1, f2)
        assert len(morph_hmm_model.morph_emit_probs) == 200
        assert round(morph_hmm_model.morph_emit_probs["1234"], 3) == round(
            0.0062799043062200955, 3
        )
        v = 1.0 / 201
        assert morph_hmm_model.translation_probs["1234"]["1234"] == v
        assert morph_hmm_model.translation_prob("1234", "1234") == v
        assert morph_hmm_model.translation_log_prob("1234", "1234") == math.log(v)
        shutil.rmtree(tmp_dir)

    def test_zero_out_params(self):
        morph_hmm_model = BilingualMorphologyHMMParams()

        tmp_dir, f1, f2 = get_two_tmp_files()
        morph_hmm_model.init_params_from_data(f1, f2)
        for morph in morph_hmm_model.translation_probs.keys():
            assert morph_hmm_model.morph_emit_probs[morph] > 0
            for target_morph in morph_hmm_model.translation_probs.keys():
                assert morph_hmm_model.translation_probs[morph][target_morph] > 0

        morph_hmm_model.zero_out_params()
        for morph in morph_hmm_model.morph_emit_probs.keys():
            assert morph_hmm_model.morph_emit_probs[morph] == 0
            for target_morph in morph_hmm_model.translation_probs.keys():
                assert morph_hmm_model.translation_probs[morph][target_morph] == 0
        shutil.rmtree(tmp_dir)

    def test_bilingual_segmentation(self):
        morph_hmm_model = BilingualMorphologyHMMParams()

        tmp_dir, f1, f2 = get_two_tmp_files()
        morph_hmm_model.init_params_from_data(f1, f2)
        segmentor = BilingualMorphologySegmentor(morph_hmm_model)
        assert segmentor.segment_viterbi("1234 1234") == [0, 2, 4, 7, 9]
        assert segmentor.segment_blingual_viterbi("1234 1234", "1234 1234") == [
            0,
            4,
            7,
            9,
        ]
        shutil.rmtree(tmp_dir)

    def test_save_load(self):
        morph_hmm_model = BilingualMorphologyHMMParams()

        tmp_dir, f1, f2 = get_two_tmp_files()
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

    def test_forward_backward(self):
        tmp_dir, f1, f2 = get_two_tmp_files()
        unsupervised_model = UnsupervisedBilingualMorphology(
            src_file=f1, dst_file=f2, smoothing_const=0.0
        )
        print(unsupervised_model.params.smoothing_const)
        # todo will add stuff here later.
        shutil.rmtree(tmp_dir)

    def test_get_morpheme_counts(self):
        tmp_dir, f1, f2 = get_two_tmp_files()
        unsupervised_model = UnsupervisedBilingualMorphology(
            src_file=f1, dst_file=f2, smoothing_const=0.0
        )
        morph_counts_with_null = unsupervised_model.params.get_morpheme_counts(
            "1234", take_log=False, include_null=True
        )
        assert len(morph_counts_with_null) == 11
        morph_counts_without_null = unsupervised_model.params.get_morpheme_counts(
            "1234", take_log=True, include_null=False
        )
        assert len(morph_counts_without_null) == 10
        assert morph_counts_without_null["12"] == math.log(morph_counts_with_null["12"])

        morph_counts_with_repeats = unsupervised_model.params.get_morpheme_counts(
            "12312", take_log=False, include_null=True
        )
        assert len(morph_counts_with_repeats) == 13
        assert morph_counts_with_repeats["12"] == 2
        shutil.rmtree(tmp_dir)
