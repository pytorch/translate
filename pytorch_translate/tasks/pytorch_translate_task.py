#!/usr/bin/env python3

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import numpy as np
from fairseq import data, options
from fairseq.data import LanguagePairDataset
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset
from fairseq.tasks import FairseqTask, register_task
from pytorch_translate import dictionary as pytorch_translate_dictionary
from pytorch_translate.data import (
    char_data,
    data as pytorch_translate_data,
    utils as data_utils,
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
            default=False,
            type=bool,
            metavar="BOOL",
            help="pad the source on the left (default: False)",
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
        args.left_pad_source = getattr(args, "left_pad_source", False)
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

        use_char_source = (
            (args.char_source_vocab_file != "")
            or (getattr(args, "arch", "") == "char_source")
            or (getattr(args, "arch", "") == "char_source_transformer")
            or getattr(args, "arch", "") == "char_source_hybrid"
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

    def _load_dataset_single_path(
        self, split: str, src_bin_path: str, tgt_bin_path: str, weights_file=None
    ):
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
            self.datasets[split] = LanguagePairDataset(
                src=src_dataset,
                src_sizes=src_dataset.sizes,
                src_dict=self.source_dictionary,
                tgt=dst_dataset,
                tgt_sizes=dst_dataset.sizes,
                tgt_dict=self.target_dictionary,
                left_pad_source=False,
            )

    def _normalized_weighted_sampling(self, weights: Dict[str, float]):
        factor = 1.0 / sum(weights.values())
        normalized_weights = {k: v * factor for k, v in weights.items()}

        def sample(candidate_list):
            v = np.random.random()
            agg = 0
            for key in candidate_list:
                agg += normalized_weights[key]
                if agg > v:
                    return key

        return sample

    def _load_dataset_multi_path(
        self,
        split: str,
        src_multiple_bin_paths: Dict[str, str],
        tgt_multiple_bin_paths: Dict[str, str],
        dataset_upsampling: Optional[Dict[str, float]],
    ):
        corpora_map = pytorch_translate_data.ParallelCorporaMapConfig(
            src_files=src_multiple_bin_paths, tgt_files=tgt_multiple_bin_paths
        )
        datasets = OrderedDict()
        for key in corpora_map.src_files:
            src, tgt = corpora_map.src_files[key], corpora_map.tgt_files[key]
            src_dataset, tgt_dataset = (
                pytorch_translate_data.InMemoryNumpyDataset.create_from_file(src),
                pytorch_translate_data.InMemoryNumpyDataset.create_from_file(tgt),
            )
            datasets[key] = LanguagePairDataset(
                src=src_dataset,
                src_sizes=src_dataset.sizes,
                src_dict=self.source_dictionary,
                tgt=tgt_dataset,
                tgt_sizes=tgt_dataset.sizes,
                tgt_dict=self.target_dictionary,
                left_pad_source=False,
            )
        dataset_weights = {
            key: 1.0 / len(src_multiple_bin_paths)
            for key in src_multiple_bin_paths.keys()
        }

        if dataset_upsampling is not None:
            for k, v in dataset_upsampling.items():
                dataset_weights[k] *= v

        self.datasets[split] = MultiCorpusSampledDataset(
            datasets=datasets,
            default_key=list(dataset_weights.keys())[0],
            sampling_func=self._normalized_weighted_sampling(dataset_weights),
        )

    def load_dataset(
        self,
        split: str,
        src_bin_path: Union[str, Dict[str, str]],
        tgt_bin_path: Union[str, Dict[str, str]],
        weights_file=None,
        dataset_upsampling: Optional[Dict[str, float]] = None,
    ):
        if type(src_bin_path) is str:
            assert type(tgt_bin_path) is str
            self._load_dataset_single_path(
                split, src_bin_path, tgt_bin_path, weights_file
            )
        else:
            assert type(tgt_bin_path) is not str
            assert set(src_bin_path.keys()) == set(tgt_bin_path.keys())
            if dataset_upsampling is not None:
                for key in dataset_upsampling.keys():
                    assert key in src_bin_path.keys()
            self._load_dataset_multi_path(
                split, src_bin_path, tgt_bin_path, dataset_upsampling
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
                src=src_dataset,
                src_sizes=src_dataset.sizes,
                src_dict=self.source_dictionary,
                tgt=dst_dataset,
                tgt_sizes=dst_dataset.sizes,
                tgt_dict=self.target_dictionary,
            )
        else:
            src_dataset = data.IndexedRawTextDataset(
                path=source_text_file,
                dictionary=self.source_dictionary,
                append_eos=append_eos,
                reverse_order=reverse_source,
            )
            self.datasets[split] = data.LanguagePairDataset(
                src=src_dataset,
                src_sizes=src_dataset.sizes,
                src_dict=self.source_dictionary,
                tgt=dst_dataset,
                tgt_sizes=dst_dataset.sizes,
                tgt_dict=self.target_dictionary,
                left_pad_source=False,
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
        tgt_dataset = data.IndexedRawTextDataset(
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
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            src_dict=self.source_dictionary,
            tgt=tgt_dataset,
            tgt_sizes=tgt_dataset.sizes,
            tgt_dict=self.target_dictionary,
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
        tgt_dataset = pytorch_translate_data.IndexedRawTextDatasetWithLangId(
            path=target_text_file,
            dictionary=self.target_dictionary,
            lang_id=target_lang_id,
            append_eos=True,
            reverse_order=False,
            prepend_language_id=True,
        )
        self.datasets[split] = data.LanguagePairDataset(
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            src_dict=self.source_dictionary,
            tgt=tgt_dataset,
            tgt_sizes=tgt_dataset.sizes,
            tgt_dict=self.target_dictionary,
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
