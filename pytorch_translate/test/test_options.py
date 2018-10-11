#!/usr/bin/env python3

import argparse
import unittest

from pytorch_translate import options
from pytorch_translate.test import utils as test_utils


class TestOptions(unittest.TestCase):
    def get_common_data_args_namespace(self):
        args = argparse.Namespace()
        args.train_source_text_file = test_utils.make_temp_file()
        args.train_target_text_file = test_utils.make_temp_file()
        args.eval_source_text_file = test_utils.make_temp_file()
        args.eval_target_text_file = test_utils.make_temp_file()
        args.source_vocab_file = test_utils.make_temp_file()
        args.target_vocab_file = test_utils.make_temp_file()
        args.task = "pytorch_translate"
        return args

    def test_validate_preprocesing_args(self):
        """ Make sure we validation passes with the minimum args required. """
        args = self.get_common_data_args_namespace()
        options.validate_preprocessing_args(args)

    def test_validate_fails_for_missing_preprocessing_arg(self):
        """
        We expect a ValueError when filepaths for a certain data type is
        missing. In this case, train source is not set at all -- no text file or
        binary path corresponds to this required data.
        """
        args = self.get_common_data_args_namespace()
        args.train_source_text_file = None
        self.assertRaises(ValueError, options.validate_preprocessing_args, args)

    def test_validate_fails_for_invalid_file(self):
        """ We expect a ValueError when a filepath is invalid """
        args = self.get_common_data_args_namespace()
        args.train_source_text_file = "nonexistent_file_path"
        self.assertRaises(ValueError, options.validate_preprocessing_args, args)

    def test_validate_preprocessing_args_monolingual(self):
        """
        Make sure we pass validation with the semisupervised training
        task when the required monolingual source and target data is
        set.
        """
        args = self.get_common_data_args_namespace()
        args.task = "pytorch_translate_semisupervised"
        args.train_mono_source_binary_path = test_utils.make_temp_file()
        args.train_mono_target_text_file = test_utils.make_temp_file()
        options.validate_preprocessing_args(args)

    def test_validate_preprocessing_args_monolingual_source_only(self):
        """
        Make sure we pass validation with the semisupervised training
        task when we only have monolingual source data.
        """
        args = self.get_common_data_args_namespace()
        args.task = "pytorch_translate_semisupervised"
        args.train_mono_source_binary_path = test_utils.make_temp_file()
        options.validate_preprocessing_args(args)

    def test_validate_preprocessing_args_monolingual_target_only(self):
        """
        Make sure we pass validation with the semisupervised training
        task when we only have monolingual source data.
        """
        args = self.get_common_data_args_namespace()
        args.task = "pytorch_translate_semisupervised"
        args.train_mono_target_binary_path = test_utils.make_temp_file()
        options.validate_preprocessing_args(args)

    def test_validate_preprocessing_args_monolingual_fails_with_missing(self):
        """
        We expect a ValueError with the semisupervised task if there's
        no monolingual data at all.
        """
        args = self.get_common_data_args_namespace()
        args.task = "pytorch_translate_semisupervised"
        self.assertRaises(ValueError, options.validate_preprocessing_args, args)
