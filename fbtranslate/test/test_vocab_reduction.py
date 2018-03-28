#!/usr/bin/env python3

import numpy as np
import unittest

from language_technology.neural_mt.fbtranslate import vocab_reduction
from language_technology.neural_mt.fbtranslate.test import utils as test_utils
from fblearner.flow.core import types
import fblearner.flow.projects.langtech.translation.common.\
    test_util as test_util_caffe2
import fblearner.flow.projects.langtech.translation.neural_mt.\
    model_util_caffe2 as model_util_caffe2


class TestVocabReduction(unittest.TestCase):
    def test_get_translation_candidates(self):
        src_dict, dst_dict = test_utils.create_caffe2_like_vocab_files()
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        translation_candidates = vocab_reduction.get_translation_candidates(
            src_dict=src_dict,
            dst_dict=dst_dict,
            lexical_dictionaries=lexical_dictionaries,
            num_top_words=10,
            max_translation_candidates_per_word=1,
        )
        (
            source_vocab_processor,
            target_vocab_processor,
            output_vocab_reduction,
        ) = test_util_caffe2.vocab_reduction_apparatus(temp_dir='/tmp')
        output_vocab_reduction = types.union.get_union_item(
            output_vocab_reduction)[1]
        translation_candidates_ref = model_util_caffe2.get_translation_candidates(
            source_vocab_processor,
            target_vocab_processor,
            output_vocab_reduction,
        )
        assert translation_candidates.size != 0
        np.testing.assert_array_equal(
            translation_candidates,
            translation_candidates_ref,
        )
