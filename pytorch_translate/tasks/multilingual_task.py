#!/usr/bin/env python3

from collections import OrderedDict

from fairseq import options
from fairseq.data import RoundRobinZipDatasets
from fairseq.tasks import register_task
from pytorch_translate import constants
from pytorch_translate.data import (
    data as pytorch_translate_data,
    utils as data_utils,
    weighted_data,
)
from pytorch_translate.tasks import utils as tasks_utils
from pytorch_translate.tasks.pytorch_translate_multi_task import (
    PyTorchTranslateMultiTask,
)


@register_task(constants.MULTILINGUAL_TRANSLATION_TASK)
class PyTorchTranslateMultilingualTranslationTask(PyTorchTranslateMultiTask):
    """
    PyTorchTranslateMultilingualTranslationTask is eventually subclasses
    fairseq.tasks.MultilingualTranslationTask. The major differences are-
    - There is no --data folder containing data binaries and vocabularies,
      instead we use paths from --vocabulary, --multilingual-*-text-file and
      --multilingual-*-binary-path
    - loss_weights is used to weigh losses from different datasets differently.
      This is achieved by using pytorch_translate's WeightedLanguagePairDataset
    - The dictionaries are instances of pytorch_translate's Dictionary class
    """

    @staticmethod
    def add_args(parser):
        PyTorchTranslateMultiTask.add_args(parser)
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "--vocabulary",
            type=str,
            metavar="EXPR",
            action="append",
            help=(
                "Per-language vocabulary configuration."
                "Path to vocabulary file must be in the format lang:path"
            ),
            default=[],
        )
        parser.add_argument(
            "--multilingual-train-text-file",
            type=str,
            metavar="EXPR",
            action="append",
            help=(
                "Path to train text file in the format "
                "src_lang-tgt_lang:source-path,target-path"
            ),
        )
        parser.add_argument(
            "--multilingual-eval-text-file",
            type=str,
            metavar="EXPR",
            action="append",
            help=(
                "Path to eval text file in the format "
                "src_lang-tgt_lang:source-path,target-path"
            ),
        )
        parser.add_argument(
            "--multilingual-train-binary-path",
            type=str,
            metavar="EXPR",
            action="append",
            help=(
                "Path to train binary file in the format "
                "src_lang-tgt_lang:source-path,target-path"
            ),
        )
        parser.add_argument(
            "--multilingual-eval-binary-path",
            type=str,
            metavar="EXPR",
            action="append",
            help=(
                "Path to eval binary file in the format "
                "src_lang-tgt_lang:source-path,target-path"
            ),
        )

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        self.loss_weights = []

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        if args.source_lang is not None or args.target_lang is not None:
            if args.lang_pairs is not None:
                raise ValueError(
                    "--source-lang/--target-lang implies generation, which is "
                    "incompatible with --lang-pairs"
                )
            training = False
            args.lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]
        else:
            training = True
            args.lang_pairs = args.lang_pairs.split(",")
            args.source_lang, args.target_lang = args.lang_pairs[0].split("-")

        dicts = tasks_utils.load_multilingual_vocabulary(args)

        return cls(args, dicts, training)

    def load_dataset(self, split, **kwargs):
        """Load a dataset split."""

        lang_pair_to_datasets = {}

        binary_path_arg = (
            "--multilingual-train-binary-path"
            if split == "train"
            else "--multilingual-eval-binary-path"
        )
        binary_path_value = (
            self.args.multilingual_train_binary_path
            if split == "train"
            else self.args.multilingual_eval_binary_path
        )

        format_warning = (
            f"{binary_path_arg} has to be in the format "
            " src_lang-tgt_lang:src_dataset_path,tgt_dataset_path"
        )

        for path_config in binary_path_value:
            # path_config: str
            # in the format "src_lang-tgt_lang:src_dataset_path,tgt_dataset_path"
            assert ":" in path_config, format_warning
            lang_pair, dataset_paths = path_config.split(":")

            assert "-" in lang_pair, format_warning

            assert "," in dataset_paths, format_warning
            src_dataset_path, tgt_dataset_path = dataset_paths.split(",")

            lang_pair_to_datasets[lang_pair] = (src_dataset_path, tgt_dataset_path)

        for lang_pair in self.args.lang_pairs:
            assert (
                lang_pair in lang_pair_to_datasets
            ), "Not all language pairs have dataset binary paths specified!"

        datasets = {}
        for lang_pair in self.args.lang_pairs:
            src, tgt = lang_pair.split("-")
            src_bin_path, tgt_bin_path = lang_pair_to_datasets[lang_pair]
            corpus = pytorch_translate_data.ParallelCorpusConfig(
                source=pytorch_translate_data.CorpusConfig(
                    dialect=src, data_file=src_bin_path
                ),
                target=pytorch_translate_data.CorpusConfig(
                    dialect=tgt, data_file=tgt_bin_path
                ),
            )
            if self.args.log_verbose:
                print("Starting to load binarized data files.", flush=True)

            data_utils.validate_corpus_exists(corpus=corpus, split=split)

            tgt_dataset = (
                pytorch_translate_data.InMemoryIndexedDataset.create_from_file(
                    corpus.target.data_file
                )
            )
            src_dataset = (
                pytorch_translate_data.InMemoryIndexedDataset.create_from_file(
                    corpus.source.data_file
                )
            )
            datasets[lang_pair] = weighted_data.WeightedLanguagePairDataset(
                src=src_dataset,
                src_sizes=src_dataset.sizes,
                src_dict=self.dicts[src],
                tgt=tgt_dataset,
                tgt_sizes=tgt_dataset.sizes,
                tgt_dict=self.dicts[tgt],
                weights=None,
                left_pad_source=False,
            )
        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict(
                [(lang_pair, datasets[lang_pair]) for lang_pair in self.args.lang_pairs]
            ),
            eval_key=None
            if self.training
            else f"{self.args.source_lang}-{self.args.target_lang}",
        )

        if self.args.log_verbose:
            print("Finished loading dataset", flush=True)

        print(f"| {split} {len(self.datasets[split])} examples")
