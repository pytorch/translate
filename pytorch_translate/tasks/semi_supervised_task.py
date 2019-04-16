#!/usr/bin/env python3

import json
from collections import OrderedDict
from typing import Optional

from fairseq import models
from fairseq.data import (
    BacktranslationDataset,
    FairseqDataset,
    LanguagePairDataset,
    RoundRobinZipDatasets,
    TransformEosDataset,
    data_utils,
)
from fairseq.models import FairseqMultiModel
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from pytorch_translate import (
    beam_decode,
    constants,
    dictionary as pytorch_translate_dictionary,
    rnn,
)
from pytorch_translate.data import (
    data as ptt_data,
    iterators as ptt_iterators,
    utils as ptt_data_utils,
    weighted_data,
)
from pytorch_translate.tasks.pytorch_translate_task import PytorchTranslateTask


@register_task(constants.SEMI_SUPERVISED_TASK)
class PytorchTranslateSemiSupervised(PytorchTranslateTask):
    def __init__(self, args, dicts, training):
        super().__init__(args, dicts[self.source_lang], dicts[self.target_lang])
        self.dicts = dicts
        self.langs = list(dicts.keys())
        self.lang_pairs = [
            f"{self.source_lang}-{self.target_lang}",
            f"{self.target_lang}-{self.source_lang}",
            (
                f"{self.source_lang}-{self.target_lang}_"
                f"{constants.MONOLINGUAL_DATA_IDENTIFIER}"
            ),
            (
                f"{self.target_lang}-{self.source_lang}_"
                f"{constants.MONOLINGUAL_DATA_IDENTIFIER}"
            ),
        ]
        self.eval_lang_pairs = [
            f"{self.source_lang}-{self.target_lang}",
            f"{self.target_lang}-{self.source_lang}",
        ]
        self.training = training
        self.remove_eos_from_source = not args.append_eos_to_source
        self.args = args
        # This is explicitly set so that we can re-use code from
        # MultilingualTranslationTask
        self.args.lang_pairs = self.lang_pairs
        self.model = None
        """
        loss_weights refers to weights given to training loss for constituent
        models for specified number of epochs. If we don't specify a model, they
        receive a weight of 1
        The format is [(epochs, {model: weight})]

        Sample input:
        [(5, {'src-tgt': 1, 'src-tgt_mono': 0}), (10, {'src-tgt': 0.5, 'src-tgt_mono': 0.5})]
        Here, we give assign weights as follows:
        For first 5 epochs, 'src-tgt' model gets weight 1, 'src-tgt_mono' gets 0
        For the next 10 epochs (till the end of training), 'src-tgt' model gets
            weight 0.5, 'src-tgt_mono' gets 0.5, the rest get 1

        """

        # default loss_weights is equal weighting for all model keys
        self.loss_weights = [
            (100, {"src-tgt": 1, "src-tgt_mono": 1, "tgt-src": 1, "tgt-src_mono": 1})
        ]
        if hasattr(self.args, "loss_weights_json") and self.args.loss_weights_json:
            self.loss_weights = PytorchTranslateSemiSupervised.parse_loss_weights(
                loss_weights_json=self.args.loss_weights_json
            )

    @staticmethod
    def add_args(parser):
        PytorchTranslateTask.add_args(parser)

        """Add semi-supervised arguments to the parser."""
        parser.add_argument(
            "--train-mono-source-binary-path",
            default="",
            help="Path for the binary file containing monolingual source "
            "training examples.",
        )
        parser.add_argument(
            "--train-mono-target-binary-path",
            default="",
            help="Path for the binary file containing monolingual target "
            "training examples.",
        )
        parser.add_argument(
            "--train-mono-source-text-file",
            default="",
            help="Path for the text file containing monolingual source "
            "training examples.",
        )
        parser.add_argument(
            "--train-mono-target-text-file",
            default="",
            help="Path for the text file containing monolingual target "
            "training examples.",
        )
        parser.add_argument(
            "--monolingual-ratio",
            default=None,
            type=float,
            metavar="N",
            help="Upper-bounds the number of monolingual examples to N times "
            "the amount of parallel data.",
        )
        parser.add_argument(
            "--loss-weights-json",
            default="",
            help="JSON representation of `loss_weights`:"
            "[[num_epochs, {'model_key': weight, ...}], ...]",
        )

    @staticmethod
    def parse_loss_weights(loss_weights_json: str):
        # [[num_epochs, {'model_key': weight, ...}], ...]
        loss_weights_decoded_json = json.loads(loss_weights_json.replace("'", '"'))
        # [(num_epochs, {'model_key': weight, ...}), ...]
        loss_weights = [
            (num_epochs, weights_dict)
            for num_epochs, weights_dict in loss_weights_decoded_json
        ]
        return loss_weights

    def load_monolingual_dataset(
        self, bin_path, is_source=False, num_examples_limit=None
    ):
        return ptt_data_utils.load_monolingual_dataset(
            bin_path=bin_path,
            is_source=is_source,
            char_source_dict=self.char_source_dict,
            log_verbose=self.args.log_verbose,
            num_examples_limit=num_examples_limit,
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        # load dictionaries
        cls.source_lang = args.source_lang or "src"
        cls.target_lang = args.target_lang or "tgt"

        dicts = OrderedDict()
        dicts[cls.source_lang] = pytorch_translate_dictionary.Dictionary.load(
            args.source_vocab_file
        )
        dicts[cls.target_lang] = pytorch_translate_dictionary.Dictionary.load(
            args.target_vocab_file
        )
        print(f"| [{cls.source_lang}] dictionary: {len(dicts[cls.source_lang])} types")
        print(f"| [{cls.target_lang}] dictionary: {len(dicts[cls.target_lang])} types")
        # TODO: Handle task setup for evals
        return cls(args, dicts, training=True)

    def load_dataset(
        self, split, src_bin_path, tgt_bin_path, forward_model=None, backward_model=None
    ):
        """Load a dataset split."""

        corpus = ptt_data.ParallelCorpusConfig(
            source=ptt_data.CorpusConfig(
                dialect=self.source_lang, data_file=src_bin_path
            ),
            target=ptt_data.CorpusConfig(
                dialect=self.target_lang, data_file=tgt_bin_path
            ),
            weights_file=None,
        )

        if self.args.log_verbose:
            print("Starting to load binarized data files.", flush=True)
        ptt_data_utils.validate_corpus_exists(corpus=corpus, split=split)

        forward_tgt_dataset = ptt_data.InMemoryNumpyDataset.create_from_file(
            corpus.target.data_file
        )
        backward_tgt_dataset = ptt_data.InMemoryNumpyDataset.create_from_file(
            corpus.source.data_file
        )
        forward_src_dataset = ptt_data.InMemoryNumpyDataset.create_from_file(
            corpus.source.data_file
        )
        backward_src_dataset = ptt_data.InMemoryNumpyDataset.create_from_file(
            corpus.target.data_file
        )
        forward_parallel_dataset = weighted_data.WeightedLanguagePairDataset(
            src=forward_src_dataset,
            src_sizes=forward_src_dataset.sizes,
            src_dict=self.source_dictionary,
            tgt=forward_tgt_dataset,
            tgt_sizes=forward_tgt_dataset.sizes,
            tgt_dict=self.target_dictionary,
            remove_eos_from_source=self.remove_eos_from_source,
            append_eos_to_target=True,
        )
        backward_parallel_dataset = weighted_data.WeightedLanguagePairDataset(
            src=backward_src_dataset,
            src_sizes=backward_src_dataset.sizes,
            src_dict=self.target_dictionary,
            tgt=backward_tgt_dataset,
            tgt_sizes=backward_tgt_dataset.sizes,
            tgt_dict=self.source_dictionary,
            remove_eos_from_source=self.remove_eos_from_source,
            append_eos_to_target=True,
        )

        dataset_map = OrderedDict(
            [
                (f"{self.source_lang}-{self.target_lang}", forward_parallel_dataset),
                (f"{self.target_lang}-{self.source_lang}", backward_parallel_dataset),
            ]
        )

        assert (forward_model and backward_model) or (
            forward_model is None and backward_model is None
        ), (
            "Only one of forward or backward models can't be null;"
            " both have to be non-null or null"
        )
        if forward_model and backward_model:
            fwd_generator = beam_decode.SequenceGenerator(
                models=[forward_model], tgt_dict=self.source_dictionary
            )
            bwd_generator = beam_decode.SequenceGenerator(
                models=[backward_model], tgt_dict=self.target_dictionary
            )

            def monolingual_dataset(
                path,
                dictionary,
                is_source=False,
                num_examples_limit: Optional[int] = None,
            ):
                dataset = self.load_monolingual_dataset(
                    path, is_source=is_source, num_examples_limit=num_examples_limit
                )
                return LanguagePairDataset(
                    src=dataset,
                    src_sizes=dataset.sizes,
                    src_dict=dictionary,
                    tgt=None,
                    tgt_sizes=None,
                    tgt_dict=None,
                )

            monolingual_num_examples_limit = None
            if self.args.monolingual_ratio is not None:
                monolingual_num_examples_limit = int(
                    self.args.monolingual_ratio * len(forward_parallel_dataset)
                )

            src_dataset = monolingual_dataset(
                path=self.args.train_mono_source_binary_path,
                dictionary=self.source_dictionary,
                is_source=True,
                num_examples_limit=monolingual_num_examples_limit,
            )
            tgt_dataset = monolingual_dataset(
                path=self.args.train_mono_target_binary_path,
                dictionary=self.target_dictionary,
                is_source=False,
                num_examples_limit=monolingual_num_examples_limit,
            )

            def generate_fn(generator):
                def _generate_fn(sample):
                    net_input = sample["net_input"]
                    maxlen = int(
                        self.args.max_len_a * net_input["src_tokens"].size(1)
                        + self.args.max_len_b
                    )
                    return generator.generate(net_input, maxlen=maxlen)

                return _generate_fn

            dataset_map[
                f"{self.source_lang}-"
                f"{self.target_lang}_{constants.MONOLINGUAL_DATA_IDENTIFIER}"
            ] = weighted_data.WeightedBacktranslationDataset(
                dataset=BacktranslationDataset(
                    tgt_dataset=TransformEosDataset(
                        dataset=tgt_dataset,
                        eos=self.target_dictionary.eos(),
                        # Remove EOS from the input before backtranslation.
                        remove_eos_from_src=True,
                    ),
                    src_dict=self.source_dictionary,
                    backtranslation_fn=generate_fn(bwd_generator),
                    output_collater=TransformEosDataset(
                        dataset=tgt_dataset,
                        eos=self.target_dictionary.eos(),
                        # The original input (now the target) doesn't have
                        # an EOS, so we need to add one. The generated
                        # backtranslation (now the source) will have an EOS,
                        # so we want to remove it.
                        append_eos_to_tgt=True,
                        remove_eos_from_src=True,
                    ).collater,
                )
            )
            dataset_map[
                f"{self.target_lang}-"
                f"{self.source_lang}_{constants.MONOLINGUAL_DATA_IDENTIFIER}"
            ] = weighted_data.WeightedBacktranslationDataset(
                dataset=BacktranslationDataset(
                    tgt_dataset=src_dataset,
                    src_dict=self.source_dictionary,
                    backtranslation_fn=generate_fn(fwd_generator),
                    output_collater=TransformEosDataset(
                        dataset=src_dataset,
                        eos=self.source_dictionary.eos(),
                        # The original input (now the target) doesn't have
                        # an EOS, so we need to add one. The generated
                        # backtranslation (now the source) will have an EOS,
                        # so we want to remove it.
                        append_eos_to_tgt=True,
                        remove_eos_from_src=True,
                    ).collater,
                )
            )

        # print before loading RoundRobinZipDatasets to help catch any bugs
        for dataset_key, dataset in dataset_map.items():
            print(f"| {split}: {dataset_key} {len(dataset)} examples in dataset")

        self.datasets[split] = RoundRobinZipDatasets(dataset_map)
        print(
            f"| {split} {len(self.datasets[split])} examples in RoundRobinZipDatasets"
        )

        if self.args.log_verbose:
            print("Finished loading dataset", flush=True)

    def build_model(self, args):
        model = models.build_model(args, self)
        self.model = model
        if not isinstance(model, FairseqMultiModel):
            raise ValueError(
                "PytorchTranslateSemiSupervised task requires a FairseqMultiModel "
                "architecture"
            )
        forward_pair = "-".join([self.source_lang, self.target_lang])
        backward_pair = "-".join([self.target_lang, self.source_lang])
        self.forward_model = model.models[forward_pair]
        self.backward_model = model.models[backward_pair]
        return model

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        return MultilingualTranslationTask.train_step(
            self, sample, model, criterion, optimizer
        )

    def valid_step(self, sample, model, criterion):
        return MultilingualTranslationTask.valid_step(self, sample, model, criterion)

    def init_logging_output(self, sample):
        return MultilingualTranslationTask.init_logging_output(sample)

    def grad_denom(self, sample_sizes, criterion):
        return MultilingualTranslationTask.grad_denom(self, sample_sizes, criterion)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        return MultilingualTranslationTask.aggregate_logging_outputs(
            self, logging_outputs, criterion
        )

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
    ):
        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        indices = data_utils.filter_by_size(
            indices,
            dataset.size,
            max_positions,
            raise_exception=(not ignore_invalid_inputs),
        )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices,
            dataset.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        return ptt_iterators.WeightedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            weights=self.loss_weights,
        )

    @property
    def source_dictionary(self):
        return self.dicts[self.source_lang]

    @property
    def target_dictionary(self):
        return self.dicts[self.target_lang]
