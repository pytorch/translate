#!/usr/bin/env python3

import math
import unittest
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

            assert len(morph_hmm_model.morph_emit_probs["prefix"]) == 28
            assert "9" not in morph_hmm_model.morph_emit_probs["prefix"]
            assert len(morph_hmm_model.morph_emit_probs["stem"]) == 42
            assert "689" not in morph_hmm_model.morph_emit_probs["stem"]
            assert len(morph_hmm_model.morph_emit_probs["suffix"]) == 29
            assert "1" not in morph_hmm_model.morph_emit_probs["suffix"]
            assert morph_hmm_model.morph_emit_probs["stem"]["1234"] == 1.1 / (42 * 1.1)

            assert morph_hmm_model.affix_trans_probs["START"]["START"] == 0
            assert morph_hmm_model.affix_trans_probs["START"]["prefix"] == 0.5
            assert morph_hmm_model.affix_trans_probs["START"]["stem"] == 0.5
            assert morph_hmm_model.affix_trans_probs["START"]["suffix"] == 0
            assert morph_hmm_model.affix_trans_probs["START"]["END"] == 0
            assert morph_hmm_model.affix_trans_probs["prefix"]["START"] == 0
            assert morph_hmm_model.affix_trans_probs["prefix"]["prefix"] == 0.5
            assert morph_hmm_model.affix_trans_probs["prefix"]["stem"] == 0.5
            assert morph_hmm_model.affix_trans_probs["prefix"]["suffix"] == 0
            assert morph_hmm_model.affix_trans_probs["prefix"]["END"] == 0
            assert morph_hmm_model.affix_trans_probs["stem"]["START"] == 0
            assert morph_hmm_model.affix_trans_probs["stem"]["prefix"] == 0
            assert morph_hmm_model.affix_trans_probs["stem"]["stem"] == 1.0 / 3
            assert morph_hmm_model.affix_trans_probs["stem"]["suffix"] == 1.0 / 3
            assert morph_hmm_model.affix_trans_probs["stem"]["END"] == 1.0 / 3
            assert morph_hmm_model.affix_trans_probs["suffix"]["START"] == 0
            assert morph_hmm_model.affix_trans_probs["suffix"]["prefix"] == 0
            assert morph_hmm_model.affix_trans_probs["suffix"]["stem"] == 0
            assert morph_hmm_model.affix_trans_probs["suffix"]["suffix"] == 0.5
            assert morph_hmm_model.affix_trans_probs["suffix"]["END"] == 0.5
            assert morph_hmm_model.affix_trans_probs["END"]["START"] == 0
            assert morph_hmm_model.affix_trans_probs["END"]["prefix"] == 0
            assert morph_hmm_model.affix_trans_probs["END"]["stem"] == 0
            assert morph_hmm_model.affix_trans_probs["END"]["suffix"] == 0
            assert morph_hmm_model.affix_trans_probs["END"]["END"] == 0

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

            assert morph_hmm_model.emission_prob("stem", "1234") == 1.1 / (42 * 1.1)
            assert morph_hmm_model.emission_prob("suffix", "1") == 0.1 / (29 * 1.1)
            assert morph_hmm_model.emission_prob("END", "1") == 0

    def test_emission_log_probs(self):
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

            assert morph_hmm_model.emission_log_probs("stem", "1234") == math.log(
                1.1 / (42 * 1.1)
            )
            assert morph_hmm_model.emission_log_probs("suffix", "1") == math.log(
                0.1 / (29 * 1.1)
            )
            assert (
                morph_hmm_model.emission_log_probs("END", "1")
                == morph_hmm_model.SMALL_CONST
            )

    def test_transition_log_probs(self):
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

            assert morph_hmm_model.transition_log_prob("stem", "END") == math.log(
                1.0 / 3
            )
            assert (
                morph_hmm_model.transition_log_prob("suffix", "START")
                == morph_hmm_model.SMALL_CONST
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
            assert segmentor.segment_viterbi("123123789") == (
                ["prefix", "prefix", "stem"],
                [0, 3, 6, 9],
            )

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
            assert segmentor.segment_viterbi("123123789") == (["stem"], [0, 9])

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
            assert segmentor.segment_word("123123789789") == "123 123 789 789"
            assert (
                segmentor.segment_word("123123789789", add_affix_symbols=True)
                == "123+ 123+ 789 +789"
            )
            assert segmentor.segment_word("123") == segmentor.segment_word(
                "123", add_affix_symbols=True
            )

    def check_emission_after_forward_backward(
        self, str, e, expected_prefixes, expected_stems, expected_suffixes
    ):
        for str in get_all_substrings(str):
            if str in expected_prefixes:
                assert e[("prefix", str)] > 0
            else:
                assert e[("prefix", str)] == 0

            if str in expected_stems:
                assert e[("stem", str)] > 0
            else:
                assert e[("stem", str)] == 0

            if str in expected_suffixes:
                assert e[("suffix", str)] > 0
            else:
                assert e[("suffix", str)] == 0

            assert e[("START", str)] == 0
            assert e[("END", str)] == 0

    def test_forward_backward(self):
        with patch("builtins.open") as mock_open:
            txt_content = ["123 12123"]
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            unsupervised_model = unsupervised_morphology.UnsupervisedMorphology(
                "no_exist_file.txt", smoothing_const=0.0
            )
            e, t = unsupervised_model.forward_backward("123")

            # checking emission parameters
            expected_prefixes = {"1"}
            expected_stems = {"12", "123", "23"}
            expected_suffixes = {"3"}
            self.check_emission_after_forward_backward(
                "123", e, expected_prefixes, expected_stems, expected_suffixes
            )

            assert t[("START", "START")] == 0
            assert t[("START", "END")] == 0
            assert t[("START", "prefix")] > 0
            assert t[("START", "stem")] > 0
            assert t[("START", "suffix")] == 0
            assert t[("END", "START")] == 0
            assert t[("END", "END")] == 0
            assert t[("END", "prefix")] == 0
            assert t[("END", "stem")] == 0
            assert t[("END", "suffix")] == 0
            assert t[("prefix", "START")] == 0
            assert t[("prefix", "END")] == 0
            assert t[("prefix", "prefix")] == 0
            assert t[("prefix", "stem")] > 0
            assert t[("prefix", "suffix")] == 0
            assert t[("stem", "START")] == 0
            assert t[("stem", "END")] > 0
            assert t[("stem", "prefix")] == 0
            assert t[("stem", "stem")] == 0
            assert t[("stem", "suffix")] > 0
            assert t[("suffix", "START")] == 0
            assert t[("suffix", "END")] > 0
            assert t[("suffix", "prefix")] == 0
            assert t[("suffix", "stem")] == 0
            assert t[("suffix", "suffix")] == 0

        def test_forward_backward_with_smoothing(self):
            """
            Making sure that the algorithm works end-to-end.
            """
            with patch("builtins.open") as mock_open:
                txt_content = ["123 12123"]
                mock_open.return_value.__enter__ = mock_open
                mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
                unsupervised_model = unsupervised_morphology.UnsupervisedMorphology(
                    "no_exist_file.txt", smoothing_const=0.0
                )
            unsupervised_model = unsupervised_morphology.UnsupervisedMorphology(
                "no_exist_file.txt", smoothing_const=0.1
            )
            e, t = unsupervised_model.forward_backward("123")

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
            e, t = unsupervised_model.forward_backward("1232345")
            expected_prefixes = {"1", "12", "123", "2", "23", "3"}
            expected_stems = {"12", "123", "23", "234", "2345", "345", "45", "34"}
            expected_suffixes = {"2", "3", "4", "5", "34", "45", "345"}
            self.check_emission_after_forward_backward(
                "1232345", e, expected_prefixes, expected_stems, expected_suffixes
            )

            assert t[("START", "START")] == 0
            assert t[("START", "END")] == 0
            assert t[("START", "prefix")] > 0
            assert t[("START", "stem")] > 0
            assert t[("START", "suffix")] == 0
            assert t[("END", "START")] == 0
            assert t[("END", "END")] == 0
            assert t[("END", "prefix")] == 0
            assert t[("END", "stem")] == 0
            assert t[("END", "suffix")] == 0
            assert t[("prefix", "START")] == 0
            assert t[("prefix", "END")] == 0
            assert t[("prefix", "prefix")] > 0
            assert t[("prefix", "stem")] > 0
            assert t[("prefix", "suffix")] == 0
            assert t[("stem", "START")] == 0
            assert t[("stem", "END")] > 0
            assert t[("stem", "prefix")] == 0
            assert t[("stem", "stem")] > 0
            assert t[("stem", "suffix")] > 0
            assert t[("suffix", "START")] == 0
            assert t[("suffix", "END")] > 0
            assert t[("suffix", "prefix")] == 0
            assert t[("suffix", "stem")] == 0
            assert t[("suffix", "suffix")] > 0

    def test_EM(self):
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
            unsupervised_model.expectation_maximization(100, 10)
