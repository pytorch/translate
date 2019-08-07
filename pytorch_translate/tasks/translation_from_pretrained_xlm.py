#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq import options, tokenizer
from fairseq.tasks import register_task
from pytorch_translate import constants
from pytorch_translate.data.masked_lm_dictionary import MaskedLMDictionary
from pytorch_translate.tasks.pytorch_translate_task import PytorchTranslateTask


@register_task("pytorch_translate_translation_from_pretrained_xlm")
class PytorchTranslateTranslationFromPretrainedXLMTask(PytorchTranslateTask):
    """
    Same as TranslationTask except use the MaskedLMDictionary class so that
    we can load data that was binarized with the MaskedLMDictionary class.

    This task should be used for the entire training pipeline when we want to
    train an NMT model from a pretrained XLM checkpoint: binarizing NMT data,
    training NMT with the pretrained XLM checkpoint, and subsequent evaluation
    of that trained model.
    """

    @staticmethod
    def add_args(parser):
        PytorchTranslateTask.add_args(parser)
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "--save-only", action="store_true", help="skip eval and only do save"
        )

    @classmethod
    def load_dictionary(cls, filename):
        """Load the masked LM dictionary from the filename

        Args:
            filename (str): the filename
        """
        return MaskedLMDictionary.load(filename)

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = MaskedLMDictionary()
        for filename in filenames:
            MaskedLMDictionary.add_file_to_dictionary(
                filename, d, tokenizer.tokenize_line, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)

        # Load dictionaries
        source_dict = MaskedLMDictionary.load(args.source_vocab_file)
        target_dict = MaskedLMDictionary.load(args.target_vocab_file)

        source_lang = args.source_lang or "src"
        target_lang = args.target_lang or "tgt"

        print(f"| [{source_lang}] dictionary: {len(source_dict)} types")
        print(f"| [{target_lang}] dictionary: {len(target_dict)} types")

        use_char_source = (args.char_source_vocab_file != "") or (
            getattr(args, "arch", "") in constants.ARCHS_FOR_CHAR_SOURCE
        )
        if use_char_source:
            char_source_dict = MaskedLMDictionary.load(args.char_source_vocab_file)
            # this attribute is used for CharSourceModel construction
            args.char_source_dict_size = len(char_source_dict)
        else:
            char_source_dict = None

        return cls(args, source_dict, target_dict, char_source_dict)
