#!/usr/bin/env python3

import os
from collections import OrderedDict

import torch
from fairseq import models
from fairseq.data import (
    BacktranslationDataset,
    FairseqDataset,
    LanguagePairDataset,
    RoundRobinZipDatasets,
    TransformEosDataset,
    data_utils,
    iterators,
)
from fairseq.models import FairseqMultiModel
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from pytorch_translate import (
    beam_decode,
    constants,
    data as ptt_data,
    data_utils as ptt_data_utils,
    dictionary as pytorch_translate_dictionary,
    rnn,
    weighted_data,
)
from pytorch_translate.tasks.pytorch_translate_task import PytorchTranslateTask


class WeightedEpochBatchIterator(iterators.EpochBatchIterator):
    def __init__(
        self,
        dataset,
        collate_fn,
        batch_sampler,
        seed=1,
        num_shards=1,
        shard_id=0,
        weights=None,
    ):
        """
        Extension of fairseq.iterators.EpochBatchIterator to use an additional
        weights structure. This weighs datasets as a function of epoch value.

        Args:
            dataset (~torch.utils.data.Dataset): dataset from which to load the data
            collate_fn (callable): merges a list of samples to form a mini-batch
            batch_sampler (~torch.utils.data.Sampler): an iterator over batches of
                indices
            seed (int, optional): seed for random number generator for
                reproducibility. Default: ``1``
            num_shards (int, optional): shard the data iterator into N
                shards. Default: ``1``
            shard_id (int, optional): which shard of the data iterator to
                return. Default: ``0``
            weights: is of the format [(epoch, {dataset: weight})]
        """
        super().__init__(dataset, collate_fn, batch_sampler, seed, num_shards, shard_id)
        self.weights = weights

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator. Default: ``True``
            fix_batches_to_gpus: ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching. Default:
                ``False``
        """
        if self.weights:
            """
            Set dataset weight based on schedule and current epoch
            """
            prev_scheduled_epochs = 0
            dataset_weights_map = None
            for schedule in self.weights:
                # schedule looks like (epoch, {dataset: weight})
                if self.epoch <= schedule[0] + prev_scheduled_epochs:
                    dataset_weights_map = schedule[1]
                    break
                prev_scheduled_epochs += schedule[0]
            # Use last weights map if weights map is not specified for the current epoch
            if dataset_weights_map is None:
                dataset_weights_map = self.weights[-1][1]
            for dataset_name in self.dataset.datasets:
                if dataset_name in dataset_weights_map:
                    assert isinstance(
                        self.dataset.datasets[dataset_name],
                        weighted_data.WeightedLanguagePairDataset,
                    ) or isinstance(
                        self.dataset.datasets[dataset_name],
                        weighted_data.WeightedBacktranslationDataset,
                    )
                    self.dataset.datasets[dataset_name].weights = [
                        dataset_weights_map[dataset_name]
                    ]
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self.epoch += 1
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle, fix_batches_to_gpus=fix_batches_to_gpus
            )
        return self._cur_epoch_itr


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
        # TODO: Generalize this to be able to use other model classes like Transformer
        self.model_cls = rnn.RNNModel
        self.remove_eos_from_source = not args.append_eos_to_source
        self.args = args
        # This is explicitly set so that we can re-use code from
        # MultilingualTranslationTask
        self.args.lang_pairs = self.lang_pairs
        self.model = None
        # TODO: Expose this as easy-to-use command line argument
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
        self.loss_weights = [
            (5, {"src-tgt": 1, "src-tgt_mono": 0, "tgt-src": 1, "tgt-src_mono": 0}),
            (5, {"src-tgt": 1, "src-tgt_mono": 0.5, "tgt-src": 1, "tgt-src_mono": 0.5}),
            (100, {"src-tgt": 1, "src-tgt_mono": 1, "tgt-src": 1, "tgt-src_mono": 1}),
        ]

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

    def load_monolingual_dataset(self, bin_path, is_source=False):
        return ptt_data_utils.load_monolingual_dataset(
            bin_path=bin_path,
            is_source=is_source,
            char_source_dict=self.char_source_dict,
            log_verbose=self.args.log_verbose,
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

            def monolingual_dataset(path, dictionary):
                dataset = self.load_monolingual_dataset(path)
                return LanguagePairDataset(
                    src=dataset,
                    src_sizes=dataset.sizes,
                    src_dict=dictionary,
                    tgt=None,
                    tgt_sizes=None,
                    tgt_dict=None,
                )

            src_dataset = monolingual_dataset(
                self.args.train_mono_source_binary_path, self.source_dictionary
            )
            tgt_dataset = monolingual_dataset(
                self.args.train_mono_target_binary_path, self.target_dictionary
            )

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
                    backtranslation_fn=bwd_generator.generate,
                    max_len_a=self.args.max_len_a,
                    max_len_b=self.args.max_len_b,
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
                    backtranslation_fn=fwd_generator.generate,
                    max_len_a=self.args.max_len_a,
                    max_len_b=self.args.max_len_b,
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
        self.datasets[split] = RoundRobinZipDatasets(dataset_map)

        if self.args.log_verbose:
            print("Finished loading dataset", flush=True)

        print(f"| {split} {len(self.datasets[split])} datasets")

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
        self.forward_model = self.model_cls(
            self, model.models[forward_pair].encoder, model.models[forward_pair].decoder
        )
        self.backward_model = self.model_cls(
            self,
            model.models[backward_pair].encoder,
            model.models[backward_pair].decoder,
        )
        return model

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        return MultilingualTranslationTask.train_step(
            self, sample, model, criterion, optimizer
        )

    def valid_step(self, sample, model, criterion):
        """
        Calculate and collect loss from all models in the task on the eval dataset
        This method is copied from MultilingualTranslationTask, except we make a
        direct reference to self.eval_lang_pairs
        """
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, {}
            for lang_pair in self.eval_lang_pairs:
                if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(
                    model.models[lang_pair], sample[lang_pair]
                )
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def init_logging_output(self, sample):
        return MultilingualTranslationTask.init_logging_output(sample)

    def grad_denom(self, sample_sizes, criterion):
        return MultilingualTranslationTask.grad_denom(self, sample_sizes, criterion)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        """
        Aggregate logging outputs for each language pair
        This method is copied from MultilingualTranslationTask, except we make a
        direct reference to self.eval_lang_pairs
        """
        agg_logging_outputs = {
            lang_pair: criterion.__class__.aggregate_logging_outputs(
                [
                    logging_output.get(lang_pair, {})
                    for logging_output in logging_outputs
                ]
            )
            for lang_pair in self.eval_lang_pairs
        }

        def sum_over_languages(key):
            return sum(
                logging_output[key] for logging_output in agg_logging_outputs.values()
            )

        # flatten logging outputs
        flat_logging_output = {
            "{}:{}".format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output["loss"] = sum_over_languages("loss")
        flat_logging_output["nll_loss"] = sum_over_languages("nll_loss")
        flat_logging_output["sample_size"] = sum_over_languages("sample_size")
        flat_logging_output["nsentences"] = sum_over_languages("nsentences")
        flat_logging_output["ntokens"] = sum_over_languages("ntokens")
        return flat_logging_output

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
        return WeightedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            weights=self.loss_weights,
        )

    @property
    def source_dictionary(self):
        return self.dicts[self.source_lang]

    @property
    def target_dictionary(self):
        return self.dicts[self.target_lang]
