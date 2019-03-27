#!/usr/bin/env python3

import math
import random
import shutil
import tempfile
import unittest
from os import path
from unittest.mock import Mock, patch

from pytorch_translate.research.unsupervised_morphology import unsupervised_morphology


def get_all_substrings(string):
    length = len(string)
    for i in range(length):
        for j in range(i + 1, length + 1):
            yield (string[i:j])


class TestUnsupervisedMorphology(unittest.TestCase):
    def test_morph_init(self):
        morph_hmm_model = unsupervised_morphology.MorphologyHMMParams()
        with patch("builtins.open") as mock_open:
            txt_content = [
                "123 124 234 345",
                "112 122 123 345",
                "123456789",
                "123456 456789",
            ]
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            morph_hmm_model.init_params_from_data("no_exist_file.txt")

            assert len(morph_hmm_model.morph_emit_probs) == 51
            assert round(morph_hmm_model.morph_emit_probs["1234"], 3) == round(
                0.014141414141414142, 3
            )

    def test_zero_out_params(self):
        morph_hmm_model = unsupervised_morphology.MorphologyHMMParams()
        with patch("builtins.open") as mock_open:
            txt_content = [
                "123 124 234 345",
                "112 122 123 345",
                "123456789",
                "123456 456789",
            ]
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            morph_hmm_model.init_params_from_data("no_exist_file.txt")
            for morph in morph_hmm_model.morph_emit_probs.keys():
                assert morph_hmm_model.morph_emit_probs[morph] > 0

            morph_hmm_model.zero_out_params()
            for morph in morph_hmm_model.morph_emit_probs.keys():
                assert morph_hmm_model.morph_emit_probs[morph] == 0

    def test_emission_probs(self):
        morph_hmm_model = unsupervised_morphology.MorphologyHMMParams()
        with patch("builtins.open") as mock_open:
            txt_content = [
                "123 124 234 345",
                "112 122 123 345",
                "123456789",
                "123456 456789",
            ]
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            morph_hmm_model.init_params_from_data("no_exist_file.txt")

            # todo add more tests
            e = 0.014141414141414142
            e_r = e * math.exp(-9)
            assert round(morph_hmm_model.emission_prob("1234"), 3) == round(e_r, 3)

    def test_emission_log_prob(self):
        morph_hmm_model = unsupervised_morphology.MorphologyHMMParams()
        with patch("builtins.open") as mock_open:
            txt_content = [
                "123 124 234 345",
                "112 122 123 345",
                "123456789",
                "123456 456789",
            ]
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            morph_hmm_model.init_params_from_data("no_exist_file.txt")

            # todo add more tests
            e = 0.014141414141414142
            e_r = e * math.exp(-9)
            assert round(morph_hmm_model.emission_log_prob("1234"), 3) == round(
                math.log(e_r), 3
            )

    def test_segment_viterbi_no_smoothing(self):
        morph_hmm_model = unsupervised_morphology.MorphologyHMMParams(
            smoothing_const=0.0
        )
        with patch("builtins.open") as mock_open:
            txt_content = [
                "123 124 234 345",
                "112 122 123 345",
                "123456789",
                "123456 456789",
            ]
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            morph_hmm_model.init_params_from_data("no_exist_file.txt")

            segmentor = unsupervised_morphology.MorphologySegmentor(morph_hmm_model)
            assert segmentor.segment_viterbi("123123789") == [0, 2, 3, 5, 6, 9]

    def test_segment_viterbi_w_smoothing(self):
        morph_hmm_model = unsupervised_morphology.MorphologyHMMParams()
        with patch("builtins.open") as mock_open:
            txt_content = [
                "123 124 234 345",
                "112 122 123 345",
                "123456789",
                "123456 456789",
            ]
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            morph_hmm_model.init_params_from_data("no_exist_file.txt")

            segmentor = unsupervised_morphology.MorphologySegmentor(morph_hmm_model)
            assert segmentor.segment_viterbi("123123789") == [0, 2, 3, 5, 6, 9]

    def test_segment_word_no_smoothing(self):
        morph_hmm_model = unsupervised_morphology.MorphologyHMMParams(
            smoothing_const=0.0
        )
        with patch("builtins.open") as mock_open:
            txt_content = [
                "123 124 234 345",
                "112 122 123 345",
                "123456789",
                "123456 456789",
            ]
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            morph_hmm_model.init_params_from_data("no_exist_file.txt")

            segmentor = unsupervised_morphology.MorphologySegmentor(morph_hmm_model)
            assert segmentor.segment_word("123123789789") == "12 3 12 3 789 789"

    def check_emission_after_forward_backward(self, str, e, expected_morphemes):
        for str in get_all_substrings(str):
            if str in expected_morphemes:
                assert e[str] > 0
            else:
                assert e[str] == 0

    def test_forward_backward(self):
        with patch("builtins.open") as mock_open:
            txt_content = ["123 12123"]
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            unsupervised_model = unsupervised_morphology.UnsupervisedMorphology(
                "no_exist_file.txt", smoothing_const=0.0
            )
            e = unsupervised_model.forward_backward("123")

            # checking emission parameters
            self.check_emission_after_forward_backward(
                "123", e, get_all_substrings("123")
            )

    def test_forward_backward_long_str(self):
        with patch("builtins.open") as mock_open:
            txt_content = [
                "123 124 234 345",
                "112 122 123 345",
                "123456789",
                "123456 456789",
            ]
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            unsupervised_model = unsupervised_morphology.UnsupervisedMorphology(
                "no_exist_file.txt", smoothing_const=0.0
            )
            e = unsupervised_model.forward_backward("1232345")
            expected_morphs = {
                "1",
                "2",
                "3",
                "4",
                "5",
                "12",
                "23",
                "34",
                "45",
                "123",
                "234",
                "345",
                "2345",
            }
            self.check_emission_after_forward_backward("1232345", e, expected_morphs)

    def test_EM_alg(self):
        txt_content = [
            "work",
            "works",
            "worked",
            "working",
            "go",
            "goes",
            "gone",
            "going",
            "do",
            "does",
            "did",
            "doing",
            "see",
            "saw",
            "seen",
            "seeing",
        ]
        # Running with forward-backward.
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            unsupervised_model = unsupervised_morphology.UnsupervisedMorphology(
                "no_exist_file.txt", smoothing_const=0.0
            )
            unsupervised_model.expectation_maximization(10, 10)

        # Running with Viterbi-EM.
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            unsupervised_model = unsupervised_morphology.UnsupervisedMorphology(
                "no_exist_file.txt", smoothing_const=0.0, use_hardEM=True
            )
            unsupervised_model.expectation_maximization(10, 10)

    def test_get_expectations_from_viterbi(self):
        with patch("builtins.open") as mock_open:
            txt_content = [
                "123 124 234 345",
                "112 122 123 345",
                "123456789",
                "123456 456789",
            ]
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))

            um = unsupervised_morphology.UnsupervisedMorphology(
                "no_exist_file.txt", smoothing_const=0.0, use_hardEM=True
            )
            assert um.segmentor.segment_viterbi("123123789") == [0, 2, 3, 5, 6, 9]
            e = um.get_expectations_from_viterbi("123123789")
            assert e["12"] == 2
            assert e["789"] == 1
            assert e["89"] == 0

    def test_save_load(self):
        with patch("builtins.open") as mock_open:
            txt_content = [
                "work",
                "works",
                "worked",
                "working",
                "go",
                "goes",
                "gone",
                "going",
                "do",
                "does",
                "did",
                "doing",
                "see",
                "saw",
                "seen",
                "seeing",
            ]
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            unsupervised_model = unsupervised_morphology.UnsupervisedMorphology(
                "no_exist_file.txt", smoothing_const=0.0
            )
            unsupervised_model.expectation_maximization(3, 2)

        test_dir = tempfile.mkdtemp()
        unsupervised_model.params.save(path.join(test_dir, "test.pickle"))

        loaded_params = unsupervised_morphology.MorphologyHMMParams.load(
            path.join(test_dir, "test.pickle")
        )

        assert (
            unsupervised_model.params.morph_emit_probs == loaded_params.morph_emit_probs
        )
        assert unsupervised_model.params.word_counts == loaded_params.word_counts
        assert (
            unsupervised_model.params.smoothing_const == loaded_params.smoothing_const
        )
        assert unsupervised_model.params.SMALL_CONST == loaded_params.SMALL_CONST
        assert unsupervised_model.params.len_cost_pow == loaded_params.len_cost_pow
        shutil.rmtree(test_dir)
