#!/usr/bin/env python3

import unittest
from unittest.mock import Mock, patch

from pytorch_translate.research.unsupervised_morphology import unsupervised_morphology


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
            assert morph_hmm_model.morph_emit_probs["stem"]["1234"] == 1.0 / 42

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
