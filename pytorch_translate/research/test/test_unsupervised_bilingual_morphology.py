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
        bilingual_segment = segmentor.segment_blingual_viterbi("1234 1234", "1234 1234")
        assert bilingual_segment[0] == [0, 4, 7, 9]
        assert bilingual_segment[1] == ["4", "4", "4"]
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

    def test_translation_marginal(self):
        tmp_dir, f1, f2 = get_two_tmp_files()
        unsupervised_model = UnsupervisedBilingualMorphology(
            src_file=f1, dst_file=f2, smoothing_const=0.0
        )
        t_marginal, dst_morph_counts = unsupervised_model.get_translation_marginal(
            "122", "123231223"
        )
        assert len(dst_morph_counts) == 34
        assert unsupervised_model.params.null_symbol in dst_morph_counts
        assert dst_morph_counts["23"] == 3
        assert len(t_marginal) == 5
        shutil.rmtree(tmp_dir)

    def test_forward_backward(self):
        tmp_dir, f1, f2 = get_two_tmp_files()
        unsupervised_model = UnsupervisedBilingualMorphology(
            src_file=f1, dst_file=f2, smoothing_const=0.0
        )
        morph_expect, translation_expect = unsupervised_model.forward_backward(
            "122", "123"
        )
        assert morph_expect["12"] > 0
        assert morph_expect["21"] == 0
        assert len(translation_expect) == 35
        assert translation_expect[("12", "3")] > 0
        assert translation_expect[("12", "32")] == 0
        assert translation_expect[("12", "122")] == 0
        assert translation_expect[("12", unsupervised_model.params.null_symbol)] > 0
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

    def test_get_expectations_from_viterbi(self):
        tmp_dir, f1, f2 = get_two_tmp_files()
        unsupervised_model = UnsupervisedBilingualMorphology(
            src_file=f1, dst_file=f2, smoothing_const=0.0, use_hardEM=True
        )
        e, t = unsupervised_model.get_expectations_from_viterbi("123", "123 124")
        assert e["123"] == 1
        assert e["23"] == 0
        assert t[("123", "2")] == 1
        assert t[("23", "2")] == 0
        assert t[("123", "123")] == 0
        shutil.rmtree(tmp_dir)

    def test_EM_algorithm(self):
        tmp_dir, f1, f2 = get_two_tmp_files()
        unsupervised_model = UnsupervisedBilingualMorphology(
            src_file=f1, dst_file=f2, smoothing_const=0.0, use_hardEM=True
        )
        unsupervised_model.expectation_maximization(f1, f2, 10, 10)
        shutil.rmtree(tmp_dir)
