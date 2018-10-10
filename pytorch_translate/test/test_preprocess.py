#!/usr/bin/env python3

import argparse
import os
import unittest

from pytorch_translate import preprocess
from pytorch_translate.test import utils as test_utils


class TestPreprocess(unittest.TestCase):
    def get_common_data_args_namespace(self):
        args = argparse.Namespace()
        source_text_file, target_text_file = test_utils.create_test_text_files()
        args.train_source_text_file = source_text_file
        args.train_target_text_file = target_text_file
        args.eval_source_text_file = source_text_file
        args.eval_target_text_file = target_text_file

        # The idea is to have these filled in during preprocessing
        args.train_source_binary_path = ""
        args.train_target_binary_path = ""
        args.eval_source_binary_path = ""
        args.eval_target_binary_path = ""

        # Required data preprocessing args
        args.append_eos_to_source = False
        args.reverse_source = True

        args.multiling_source_lang = None  # Indicates no multilingual data
        args.penalized_target_tokens_file = ""

        args.source_vocab_file = test_utils.make_temp_file()
        args.source_max_vocab_size = None
        args.target_vocab_file = test_utils.make_temp_file()
        args.target_max_vocab_size = None
        args.char_source_vocab_file = ""
        return args

    def test_preprocess(self):
        """
        This is just a correctness test to make sure no errors are thrown when
        all the required args are passed. Actual parsing code is tested by
        test_data.py
        """
        args = self.get_common_data_args_namespace()
        preprocess.preprocess_corpora(args)
        for file_type in (
            "train_source_binary_path",
            "train_target_binary_path",
            "eval_source_binary_path",
            "eval_target_binary_path",
        ):
            file = getattr(args, file_type)
            assert file and os.path.isfile(file)
            assert file.endswith(".npz")
