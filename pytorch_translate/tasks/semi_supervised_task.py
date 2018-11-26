#!/usr/bin/env python3

import os
from collections import OrderedDict

import torch
from fairseq import models
from fairseq.data import (
    BacktranslationDataset,
    LanguagePairDataset,
    RoundRobinZipDatasets,
    TransformEosDataset,
)
from fairseq.models import FairseqMultiModel
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from pytorch_translate import (
    beam_decode,
    constants,
    data as ptt_data,
    data_utils,
    dictionary as pytorch_translate_dictionary,
    rnn,
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
        # TODO: Generalize this to be able to use other model classes like Transformer
        self.model_cls = rnn.RNNModel
        self.remove_eos_from_source = not args.append_eos_to_source
        self.args = args
        # This is explicitly set so that we can re-use code from
        # MultilingualTranslationTask
        self.args.lang_pairs = self.lang_pairs
        self.model = None

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
        return data_utils.load_monolingual_dataset(
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
        data_utils.validate_corpus_exists(corpus=corpus, split=split)

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
            ] = BacktranslationDataset(
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
            dataset_map[
                f"{self.target_lang}-"
                f"{self.source_lang}_{constants.MONOLINGUAL_DATA_IDENTIFIER}"
            ] = BacktranslationDataset(
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

    @property
    def source_dictionary(self):
        return self.dicts[self.source_lang]

    @property
    def target_dictionary(self):
        return self.dicts[self.target_lang]
