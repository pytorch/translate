#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os

from fairseq import tokenizer
from fairseq.tasks import register_task
from fairseq.tasks.cross_lingual_lm import CrossLingualLMTask
from pytorch_translate.data.masked_lm_dictionary import MaskedLMDictionary


@register_task("pytorch_translate_cross_lingual_lm")
class PytorchTranslateCrossLingualLMTask(CrossLingualLMTask):
    """
    Task for training cross-lingual language models.
    For more details look at: https://arxiv.org/pdf/1901.07291.pdf
    Args:
        dictionary (MaskedLMDictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        CrossLingualLMTask.add_args(parser)
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "-s", "--source-lang", default=None, metavar="SRC", help="source language"
        )
        parser.add_argument(
            "-t",
            "--target-lang",
            default=None,
            metavar="TARGET",
            help="target language",
        )
        parser.add_argument(
            "--save-only", action="store_true", help="skip eval and only do save"
        )

    @classmethod
    def load_dictionary(cls, filename):
        return MaskedLMDictionary.load(filename)

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        d = MaskedLMDictionary()
        for filename in filenames:
            MaskedLMDictionary.add_file_to_dictionary(
                filename, d, tokenizer.tokenize_line, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        if getattr(args, "raw_text", False):
            args.dataset_impl = "raw"
        elif getattr(args, "lazy_load", False):
            args.dataset_impl = "lazy"

        dictionary = MaskedLMDictionary.load(
            os.path.join(args.data, args.source_vocab_file)
        )

        print("| dictionary: {} types".format(len(dictionary)))

        return cls(args, dictionary)
