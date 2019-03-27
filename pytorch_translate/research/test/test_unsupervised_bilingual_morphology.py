#!/usr/bin/env python3

import unittest
from unittest.mock import Mock, patch

from pytorch_translate.research.unsupervised_morphology import (
    unsupervised_bilingual_morphology,
)


class TestUnsupervisedBilingualMorphology(unittest.TestCase):
    def test_morph_init(self):
        morph_hmm_model = (
            unsupervised_bilingual_morphology.BilingualMorphologyHMMParams()
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

            assert len(morph_hmm_model.alignment_probs) == 9
            assert round(morph_hmm_model.morph_emit_probs["1234"], 3) == round(
                0.014141414141414142, 3
            )
