#!/usr/bin/env python3

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
from fairseq import data, options
from fairseq.data import ConcatDataset, LanguagePairDataset, NoisingDataset
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset
from fairseq.data.noising import UnsupervisedMTNoising
from fairseq.tasks import LegacyFairseqTask, register_task
from pytorch_translate import constants, utils as pytorch_translate_utils
from pytorch_translate.data import (
    char_data,
    data as pytorch_translate_data,
    dictionary as pytorch_translate_dictionary,
    utils as data_utils,
    weighted_data,
)
from pytorch_translate.data.language_pair_upsampling_dataset import (
    LanguagePairUpsamplingDataset,
)
from pytorch_translate.research.multisource import multisource_data


@register_task("pytorch_translate")
class PytorchTranslateTask(LegacyFairseqTask):
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
        parser.add_argument(
            "--word-dropout-prob-map",
            default=None,
            help="Use NoisingDataset, and this argument specifies "
            "the probability a token is dropped randomly",
        )
        parser.add_argument(
            "--word-blank-prob-map",
            default=None,
            help="Use NoisingDataset, and this argument specifies "
            "the probability a token is replaced by unk",
        )
        parser.add_argument(
            "--max-word-shuffle-distance-map",
            default=None,
            help="Use NoisingDataset, and this argument specifies "
            "the maximum distance a word could move during the shuffle",
        )
        parser.add_argument(
            "--dataset-upsampling",
            default=None,
            metavar="FILE",
            help="Upsampling for certain datasets, with upsampling rate "
            "represented in a dictionary (dataset, rate). sampling ratio = "
            "upsampling rate * number of lines of the dataset / "
            "(upsampling rate * number of lines of the dataset"
            "+ number of total lines of other datsets). At most one of "
            "dataset_upsampling / dataset_relative_ratio could be specified.",
        )
        parser.add_argument(
            "--dataset-relative-ratio",
            default=None,
            metavar="FILE",
            help="Relative ratio(one-vs-rest) for certain dataset, "
            "represented in (dataset, ratio) tuple. It would be the final sampling"
            "ratio for certain dataset. For example when r = 0.5, half of training"
            "corpus would come from this dataset. At most one of "
            "dataset_upsampling / dataset_relative_ratio could be specified.",
        )

    def __init__(
        self, args, src_dict, tgt_dict, char_source_dict=None, char_target_dict=None
    ):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.char_source_dict = char_source_dict
        self.char_target_dict = char_target_dict

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

        use_char_source = (args.char_source_vocab_file != "") or (
            getattr(args, "arch", "") in constants.ARCHS_FOR_CHAR_SOURCE
        )
        if use_char_source:
            char_source_dict = pytorch_translate_dictionary.Dictionary.load(
                args.char_source_vocab_file
            )
            # this attribute is used for CharSourceModel construction
            args.char_source_dict_size = len(char_source_dict)
        else:
            char_source_dict = None

        use_char_target = (getattr(args, "char_target_vocab_file", "") != "") or (
            getattr(args, "arch", "") in constants.ARCHS_FOR_CHAR_TARGET
        )
        if use_char_target:
            char_target_dict = pytorch_translate_dictionary.Dictionary.load(
                args.char_target_vocab_file
            )
            args.char_target_dict_size = len(char_target_dict)
        else:
            char_target_dict = None

        return cls(
            args,
            src_dict=source_dict,
            tgt_dict=target_dict,
            char_source_dict=char_source_dict,
            char_target_dict=char_target_dict,
        )

    def _load_dataset_single_path(
        self,
        split: str,
        src_bin_path: str,
        tgt_bin_path: str,
        weights_file=None,
        is_npz=True,
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

        if getattr(self.args, "log_verbose", False):
            print("Starting to load binarized data files.", flush=True)
        append_bos = getattr(self.args, "append_bos", False)
        data_utils.validate_corpus_exists(corpus=corpus, split=split, is_npz=is_npz)

        if self.char_target_dict is not None:
            dst_dataset = char_data.InMemoryNumpyWordCharDataset.create_from_file(
                corpus.target.data_file
            )
        else:
            dst_dataset = (
                pytorch_translate_data.InMemoryIndexedDataset.create_from_file(
                    corpus.target.data_file, is_npz=is_npz
                )
            )

        if getattr(self.args, "reverse_target", None):
            dst_dataset.reverse()
        weights_dataset = None
        if corpus.weights_file and os.path.exists(corpus.weights_file):
            weights_dataset = weighted_data.IndexedWeightsDataset(corpus.weights_file)
            print(
                f"dst: {len(dst_dataset)} lines, weights: {len(weights_dataset)} lines."
            )
            assert len(dst_dataset) == len(weights_dataset)

        if self.char_source_dict is not None:
            src_dataset = char_data.InMemoryNumpyWordCharDataset.create_from_file(
                corpus.source.data_file
            )
            char_data_class = (
                char_data.LanguagePairCharDataset
                if self.char_target_dict is not None
                else char_data.LanguagePairSourceCharDataset
            )
            self.datasets[split] = char_data_class(
                src=src_dataset,
                src_sizes=src_dataset.sizes,
                src_dict=self.source_dictionary,
                tgt=dst_dataset,
                tgt_sizes=dst_dataset.sizes,
                tgt_dict=self.target_dictionary,
                weights=weights_dataset,
            )
        else:
            src_dataset = (
                pytorch_translate_data.InMemoryIndexedDataset.create_from_file(
                    corpus.source.data_file, is_npz=is_npz
                )
            )
            if getattr(self.args, "train_weights_path", None):
                self.datasets[split] = weighted_data.WeightedLanguagePairDataset(
                    src=src_dataset,
                    src_sizes=src_dataset.sizes,
                    src_dict=self.source_dictionary,
                    tgt=dst_dataset,
                    tgt_sizes=dst_dataset.sizes,
                    tgt_dict=self.target_dictionary,
                    weights=weights_dataset,
                    left_pad_source=False,
                )
            else:
                self.datasets[split] = LanguagePairDataset(
                    src=src_dataset,
                    src_sizes=src_dataset.sizes,
                    src_dict=self.source_dictionary,
                    tgt=dst_dataset,
                    tgt_sizes=dst_dataset.sizes,
                    tgt_dict=self.target_dictionary,
                    append_bos=append_bos,
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

    def _load_dataset_multi_path_helper(
        self,
        split: str,
        src_multiple_bin_paths: Dict[str, str],
        tgt_multiple_bin_paths: Dict[str, str],
        dataset_upsampling: Optional[Dict[str, float]] = None,
        dataset_relative_ratio: Optional[Tuple[str, float]] = None,
        seed: Optional[int] = None,
        noiser: Optional[Dict[str, UnsupervisedMTNoising]] = None,
        is_npz: bool = True,
    ):
        corpora_map = pytorch_translate_data.ParallelCorporaMapConfig(
            src_files=src_multiple_bin_paths, tgt_files=tgt_multiple_bin_paths
        )
        datasets = OrderedDict()
        for key in corpora_map.src_files:
            src, tgt = corpora_map.src_files[key], corpora_map.tgt_files[key]
            if self.char_target_dict is not None:
                tgt_dataset = char_data.InMemoryNumpyWordCharDataset.create_from_file(
                    tgt
                )
            else:
                tgt_dataset = (
                    pytorch_translate_data.InMemoryIndexedDataset.create_from_file(
                        tgt, is_npz=is_npz
                    )
                )

            if self.char_source_dict is not None:
                src_dataset = char_data.InMemoryNumpyWordCharDataset.create_from_file(
                    src
                )
            else:
                src_dataset = (
                    pytorch_translate_data.InMemoryIndexedDataset.create_from_file(
                        src, is_npz=is_npz
                    )
                )
            src_sizes = src_dataset.sizes
            if noiser is not None and key in noiser:
                src_dataset = NoisingDataset(
                    src_dataset=src_dataset,
                    src_dict=self.source_dictionary,
                    seed=seed,
                    noiser=noiser[key],
                )
            if self.char_source_dict is not None:
                char_data_class = (
                    char_data.LanguagePairCharDataset
                    if self.char_target_dict is not None
                    else char_data.LanguagePairSourceCharDataset
                )
                datasets[key] = char_data_class(
                    src=src_dataset,
                    src_sizes=src_sizes,
                    src_dict=self.source_dictionary,
                    tgt=tgt_dataset,
                    tgt_sizes=tgt_dataset.sizes,
                    tgt_dict=self.target_dictionary,
                )
            else:
                datasets[key] = LanguagePairDataset(
                    src=src_dataset,
                    src_sizes=src_sizes,
                    src_dict=self.source_dictionary,
                    tgt=tgt_dataset,
                    tgt_sizes=tgt_dataset.sizes,
                    tgt_dict=self.target_dictionary,
                    left_pad_source=False,
                )
        total_line_count = sum(len(datasets[key]) for key in datasets)
        if dataset_relative_ratio:
            ds, ratio = dataset_relative_ratio
            line_count = len(datasets[ds])
            # By definition ratio = u * line_count / sum(#lines of other datasets)
            u = (total_line_count - line_count) / line_count * ratio
            dataset_upsampling = {key: u}
        elif not dataset_upsampling:
            dataset_upsampling = {}

        print(f"|dataset upsampling:{dataset_upsampling}")
        ds_list = []
        sample_ratios = []
        for key, val in datasets.items():
            ds_list.append(val)
            sample_ratios.append(int(dataset_upsampling.get(key, 1)))

        self.datasets[split] = LanguagePairUpsamplingDataset(
            datasets=datasets.values(), sample_ratios=sample_ratios
        )

    def _load_dataset_multi_path(
        self, split: str, src_bin_path: str, tgt_bin_path: str, is_npz: bool = True
    ):
        assert type(tgt_bin_path) is not str
        assert set(src_bin_path.keys()) == set(tgt_bin_path.keys())
        source_lang = self.args.source_lang or "src"
        target_lang = self.args.target_lang or "tgt"
        direction = source_lang + "-" + target_lang
        dataset_upsampling = (
            pytorch_translate_utils.maybe_parse_collection_argument(
                self.args.dataset_upsampling
            )[direction]
            if self.args.dataset_upsampling
            else None
        )
        dataset_relative_ratio = (
            pytorch_translate_utils.maybe_parse_collection_argument(
                self.args.dataset_relative_ratio
            )[direction]
            if self.args.dataset_relative_ratio
            else None
        )
        noiser = {}
        noise_options = [
            "word_dropout_prob",
            "max_word_shuffle_distance",
            "word_blanking_prob",
        ]
        for option in noise_options:
            option_map = getattr(self.args, option + "_map", None)
            if option_map:
                option_map = pytorch_translate_utils.maybe_parse_collection_argument(
                    option_map
                )[direction]
                for key in option_map:
                    if key not in noiser:
                        noiser[key] = {
                            noise_option: None for noise_option in noise_options
                        }
                    noiser[key][option] = option_map[key]

        for key in noiser:
            noiser[key] = UnsupervisedMTNoising(
                dictionary=self.src_dict,
                max_word_shuffle_distance=noiser[key]["max_word_shuffle_distance"] or 0,
                word_dropout_prob=noiser[key]["word_dropout_prob"] or 0,
                word_blanking_prob=noiser[key]["word_blanking_prob"] or 0,
            )

        if dataset_relative_ratio is not None:
            assert dataset_upsampling is None, "dataset_upsampling and "
            "dataset_relative_ratio couldn't be specified together."
            assert dataset_relative_ratio[0] in src_bin_path.keys()
            self._load_dataset_multi_path_helper(
                split=split,
                src_multiple_bin_paths=src_bin_path,
                tgt_multiple_bin_paths=tgt_bin_path,
                dataset_relative_ratio=dataset_relative_ratio,
                seed=self.args.seed,
                noiser=noiser,
                is_npz=is_npz,
            )
        elif dataset_upsampling is not None:
            for key in dataset_upsampling.keys():
                assert key in src_bin_path.keys()
            self._load_dataset_multi_path_helper(
                split=split,
                src_multiple_bin_paths=src_bin_path,
                tgt_multiple_bin_paths=tgt_bin_path,
                dataset_upsampling=dataset_upsampling,
                seed=self.args.seed,
                noiser=noiser,
                is_npz=is_npz,
            )
        else:
            self._load_dataset_multi_path_helper(
                split=split,
                src_multiple_bin_paths=src_bin_path,
                tgt_multiple_bin_paths=tgt_bin_path,
                seed=self.args.seed,
                noiser=noiser,
                is_npz=is_npz,
            )

    def load_dataset(
        self,
        split: str,
        src_bin_path: str,
        tgt_bin_path: str,
        weights_file=None,
        is_npz=True,
    ):
        src_bin_path = pytorch_translate_utils.maybe_parse_collection_argument(
            src_bin_path
        )
        tgt_bin_path = pytorch_translate_utils.maybe_parse_collection_argument(
            tgt_bin_path
        )
        # At most one of dataset_upsampling / dataset_relative_ratio could be
        # specified.
        if type(src_bin_path) is str:
            assert type(tgt_bin_path) is str
            self._load_dataset_single_path(
                split=split,
                src_bin_path=src_bin_path,
                tgt_bin_path=tgt_bin_path,
                weights_file=weights_file,
                is_npz=is_npz,
            )
        else:
            self._load_dataset_multi_path(
                split, src_bin_path, tgt_bin_path, is_npz=is_npz
            )

        if getattr(self.args, "log_verbose", False):
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
        append_bos = getattr(self.args, "append_bos", False)
        if self.char_target_dict is not None:
            dst_dataset = char_data.InMemoryNumpyWordCharDataset()
            dst_dataset.parse(
                path=target_text_file,
                word_dict=self.target_dictionary,
                char_dict=self.char_target_dict,
                reverse_order=False,
                append_eos=True,
            )
        else:
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
            char_data_class = (
                char_data.LanguagePairCharDataset
                if self.char_target_dict is not None
                else char_data.LanguagePairSourceCharDataset
            )
            self.datasets[split] = char_data_class(
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
                append_bos=append_bos,
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
class DictionaryHolderTask(LegacyFairseqTask):
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
