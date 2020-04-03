#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

from fairseq import options
from fairseq.tasks import register_task
from fairseq.tasks.translation_lev import TranslationLevenshteinTask
from pytorch_translate.data import dictionary as pytorch_translate_dictionary
from pytorch_translate.tasks.pytorch_translate_task import PytorchTranslateTask


@register_task("ptt_translation_lev")
class PytorchTranslationLevenshteinTask(PytorchTranslateTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    def __init__(
        self, args, src_dict, tgt_dict, char_source_dict=None, char_target_dict=None
    ):
        super().__init__(args, src_dict, tgt_dict, char_source_dict, char_target_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.char_source_dict = char_source_dict
        self.char_target_dict = char_target_dict
        self.trans_lev_task = TranslationLevenshteinTask(args, src_dict, tgt_dict)

    @staticmethod
    def add_args(parser):
        TranslationLevenshteinTask.add_args(parser)

    def inject_noise(self, target_tokens):
        return self.trans_lev_task.inject_noise(target_tokens)

    def build_generator(self, models, args):
        self.trans_lev_task.build_generator(models, args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        source_dict = pytorch_translate_dictionary.Dictionary.load(
            args.source_vocab_file
        )
        target_dict = pytorch_translate_dictionary.Dictionary.load(
            args.target_vocab_file
        )
        source_lang = args.source_lang or "src"
        target_lang = args.target_lang or "tgt"
        args.append_bos = True

        print(f"| [{source_lang}] dictionary: {len(source_dict)} types")
        print(f"| [{target_lang}] dictionary: {len(target_dict)} types")

        return cls(args, source_dict, target_dict)

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        return self.trans_lev_task.train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )

    def valid_step(self, sample, model, criterion):
        return self.trans_lev_task.valid_step(sample, model, criterion)
