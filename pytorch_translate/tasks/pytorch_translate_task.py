#!/usr/bin/env python3

import os
from collections import OrderedDict
from typing import List, Optional

from fairseq import data, options
from fairseq.tasks import FairseqTask, register_task
from pytorch_translate import (
    char_data,
    data as pytorch_translate_data,
    data_utils,
    dictionary as pytorch_translate_dictionary,
    weighted_data,
)
from pytorch_translate.research.multisource import multisource_data


@register_task("pytorch_translate")
class PytorchTranslateTask(FairseqTask):
    @staticmethod
    def add_args(parser):
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
            "--left-pad-source",
            default="True",
            type=str,
            metavar="BOOL",
            help="pad the source on the left (default: True)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )

    def __init__(self, args, src_dict, tgt_dict, char_source_dict=None):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.char_source_dict = char_source_dict

    def build_model(self, args):
        # set defaults for old model checkpoints
        args.left_pad_source = getattr(args, "left_pad_source", True)
        return super().build_model(args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)

        assert not pytorch_translate_data.is_multilingual(
            args
        ), "Must set `--task pytorch_translate_multilingual` for multilingual training"

        # Load dictionaries
        source_dict = pytorch_translate_dictionary.Dictionary.load(
            args.source_vocab_file
        )
        target_dict = pytorch_translate_dictionary.Dictionary.load(
            args.target_vocab_file
        )

        source_lang = args.source_lang or "src"
        target_lang = args.target_lang or "tgt"

        print(f"| [{source_lang}] dictionary: {len(source_dict)} types")
        print(f"| [{target_lang}] dictionary: {len(target_dict)} types")

        use_char_source = (args.char_source_vocab_file != "") or (
            getattr(args, "arch", "") == "char_source"
        )
        if use_char_source:
            char_source_dict = pytorch_translate_dictionary.Dictionary.load(
                args.char_source_vocab_file
            )
            # this attribute is used for CharSourceModel construction
            args.char_source_dict_size = len(char_source_dict)
        else:
            char_source_dict = None

        return cls(args, source_dict, target_dict, char_source_dict)

    def load_dataset(self, split, src_bin_path, tgt_bin_path, weights_file=None):
        corpus = pytorch_translate_data.ParallelCorpusConfig(
            source=pytorch_translate_data.CorpusConfig(
                dialect=self.args.source_lang, data_file=src_bin_path
            ),
            target=pytorch_translate_data.CorpusConfig(
                dialect=self.args.target_lang, data_file=tgt_bin_path
            ),
            weights_file=weights_file,
        )

        if self.args.log_verbose:
            print("Starting to load binarized data files.", flush=True)
        data_utils.validate_corpus_exists(corpus=corpus, split=split)

        dst_dataset = pytorch_translate_data.InMemoryNumpyDataset.create_from_file(
            corpus.target.data_file
        )
        weights_dataset = None
        if corpus.weights_file and os.path.exists(corpus.weights_file):
            weights_dataset = weighted_data.IndexedWeightsDataset(corpus.weights_file)
            assert len(dst_dataset) == len(weights_dataset)

        if self.char_source_dict is not None:
            src_dataset = char_data.InMemoryNumpyWordCharDataset.create_from_file(
                corpus.source.data_file
            )
            self.datasets[split] = char_data.LanguagePairSourceCharDataset(
                src=src_dataset,
                src_sizes=src_dataset.sizes,
                src_dict=self.source_dictionary,
                tgt=dst_dataset,
                tgt_sizes=dst_dataset.sizes,
                tgt_dict=self.target_dictionary,
                weights=weights_dataset,
            )
        else:
            src_dataset = pytorch_translate_data.InMemoryNumpyDataset.create_from_file(
                corpus.source.data_file
            )
            self.datasets[split] = weighted_data.WeightedLanguagePairDataset(
                src=src_dataset,
                src_sizes=src_dataset.sizes,
                src_dict=self.source_dictionary,
                tgt=dst_dataset,
                tgt_sizes=dst_dataset.sizes,
                tgt_dict=self.target_dictionary,
                weights=weights_dataset,
            )

        if self.args.log_verbose:
            print("Finished loading dataset", flush=True)

        print(f"| {split} {len(self.datasets[split])} examples")

    def load_dataset_from_text(
        self,
        split: str,
        source_text_file: str,
        target_text_file: str,
        append_eos: Optional[bool] = False,
        reverse_source: Optional[bool] = True,
    ):
        dst_dataset = data.IndexedRawTextDataset(
            path=target_text_file,
            dictionary=self.target_dictionary,
            # We always append EOS to the target sentence since we still want
            # the model to output an indication the sentence has finished, even
            # if we don't append the EOS symbol to the source sentence
            # (to prevent the model from misaligning UNKs or other words
            # to the frequently occurring EOS).
            append_eos=True,
            # We don't reverse the order of the target sentence, since
            # even if the source sentence is fed to the model backwards,
            # we still want the model to start outputting from the first word.
            reverse_order=False,
        )

        if self.char_source_dict is not None:
            src_dataset = char_data.InMemoryNumpyWordCharDataset()
            src_dataset.parse(
                path=source_text_file,
                word_dict=self.source_dictionary,
                char_dict=self.char_source_dict,
                reverse_order=reverse_source,
                append_eos=append_eos,
            )
            self.datasets[split] = char_data.LanguagePairSourceCharDataset(
                src_dataset,
                src_dataset.sizes,
                self.source_dictionary,
                dst_dataset,
                dst_dataset.sizes,
                self.target_dictionary,
            )
        else:
            src_dataset = data.IndexedRawTextDataset(
                path=source_text_file,
                dictionary=self.source_dictionary,
                append_eos=append_eos,
                reverse_order=reverse_source,
            )
            self.datasets[split] = data.LanguagePairDataset(
                src_dataset,
                src_dataset.sizes,
                self.source_dictionary,
                dst_dataset,
                dst_dataset.sizes,
                self.target_dictionary,
            )

        print(f"| {split} {len(self.datasets[split])} examples")

    def load_multisource_dataset_from_text(
        self,
        split: str,
        source_text_files: List[str],
        target_text_file: str,
        append_eos: Optional[bool] = False,
        reverse_source: Optional[bool] = True,
    ):
        src_dataset = multisource_data.IndexedRawTextMultisentDataset(
            path=source_text_files,
            dictionary=self.source_dictionary,
            append_eos=append_eos,
            reverse_order=reverse_source,
        )
        dst_dataset = data.IndexedRawTextDataset(
            path=target_text_file,
            dictionary=self.target_dictionary,
            # We always append EOS to the target sentence since we still want
            # the model to output an indication the sentence has finished, even
            # if we don't append the EOS symbol to the source sentence
            # (to prevent the model from misaligning UNKs or other words
            # to the frequently occurring EOS).
            append_eos=True,
            # We don't reverse the order of the target sentence, since
            # even if the source sentence is fed to the model backwards,
            # we still want the model to start outputting from the first word.
            reverse_order=False,
        )
        self.datasets[split] = multisource_data.MultisourceLanguagePairDataset(
            src_dataset,
            src_dataset.sizes,
            self.source_dictionary,
            dst_dataset,
            dst_dataset.sizes,
            self.target_dictionary,
        )

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict


# We don't @register_task since this is mostly used for unit tests and export
class DictionaryHolderTask(FairseqTask):
    """A simplified Task that just holds the dictionaries."""

    def __init__(self, src_dict, dst_dict):
        super().__init__(args=None)
        self.src_dict = src_dict
        self.dst_dict = dst_dict

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.dst_dict


@register_task("pytorch_translate_multilingual")
class PytorchTranslateMultilingualTask(PytorchTranslateTask):
    def __init__(self, args, source_dictionaries, target_dictionaries):
        self.source_dictionaries = source_dictionaries
        self.target_dictionaries = target_dictionaries

        # Mapping from language IDs to language codes. During training
        # this list is fully populated. During generation we typically
        # have only a single source/target dictionary, thus it is important to
        # call set_encoder/decoder_langs to properly populate these.
        self.encoder_langs = list(source_dictionaries.keys())
        self.decoder_langs = list(target_dictionaries.keys())

        self.src_dict = pytorch_translate_dictionary.MaxVocabDictionary()
        for d in source_dictionaries.values():
            self.src_dict.push(d)
        self.tgt_dict = pytorch_translate_dictionary.MaxVocabDictionary()
        for d in target_dictionaries.values():
            self.tgt_dict.push(d)

        super().__init__(args, self.src_dict, self.tgt_dict)

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert pytorch_translate_data.is_multilingual(
            args
        ), "Must set `--task pytorch_translate_multilingual` for multilingual training"
        args.left_pad_source = options.eval_bool(args.left_pad_source)

        def load_dicts(langs, paths):
            dicts = OrderedDict()
            for lang, dict_path in zip(langs, paths):
                d = pytorch_translate_dictionary.Dictionary.load(dict_path)
                dicts[lang] = d
                print(f"| [{lang}] dictionary: {len(d)} types")
            return dicts

        if not hasattr(args, "multiling_source_vocab_file"):
            args.multiling_encoder_lang = args.multiling_source_lang
            args.multiling_source_vocab_file = [args.source_vocab_file]
        if not hasattr(args, "multiling_target_vocab_file"):
            args.multiling_decoder_lang = args.multiling_target_lang
            args.multiling_target_vocab_file = [args.target_vocab_file]

        # Load dictionaries
        src_dicts = load_dicts(
            args.multiling_encoder_lang, args.multiling_source_vocab_file
        )
        tgt_dicts = load_dicts(
            args.multiling_decoder_lang, args.multiling_target_vocab_file
        )

        return cls(args, src_dicts, tgt_dicts)

    def load_dataset_from_text_multilingual(
        self,
        split: str,
        source_text_file: str,
        target_text_file: str,
        source_lang_id: int,
        target_lang_id: int,
        append_eos: Optional[bool] = False,
        reverse_source: Optional[bool] = True,
    ):
        src_dataset = pytorch_translate_data.IndexedRawTextDatasetWithLangId(
            path=source_text_file,
            dictionary=self.source_dictionary,
            lang_id=source_lang_id,
            append_eos=append_eos,
            reverse_order=reverse_source,
            prepend_language_id=False,
        )
        dst_dataset = pytorch_translate_data.IndexedRawTextDatasetWithLangId(
            path=target_text_file,
            dictionary=self.target_dictionary,
            lang_id=target_lang_id,
            append_eos=True,
            reverse_order=False,
            prepend_language_id=True,
        )
        self.datasets[split] = data.LanguagePairDataset(
            src_dataset,
            src_dataset.sizes,
            self.source_dictionary,
            dst_dataset,
            dst_dataset.sizes,
            self.target_dictionary,
        )
        print(f"| {split} {len(self.datasets[split])} examples")

    def set_encoder_langs(self, encoder_langs):
        self.encoder_langs = encoder_langs

    def set_decoder_langs(self, decoder_langs):
        self.decoder_langs = decoder_langs

    def get_encoder_lang_id(self, lang_code):
        return self.encoder_langs.index(lang_code)

    def get_decoder_lang_id(self, lang_code):
        return self.decoder_langs.index(lang_code)

    def get_encoder_lang_code(self, lang_id):
        return self.encoder_langs[lang_id]

    def get_decoder_lang_code(self, lang_id):
        return self.decoder_langs[lang_id]
