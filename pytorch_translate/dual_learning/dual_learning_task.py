#!/usr/bin/env python3

from collections import OrderedDict

import torch
from fairseq import optim, utils
from fairseq.criterions import CRITERION_REGISTRY
from fairseq.data import LanguagePairDataset, RoundRobinZipDatasets
from fairseq.tasks import FairseqTask, register_task
from pytorch_translate import dictionary as pytorch_translate_dictionary
from pytorch_translate.data import utils as data_utils
from pytorch_translate.dual_learning import dual_learning_models
from pytorch_translate.dual_learning.dual_learning_criterion import (
    UnsupervisedCriterion,
)
from pytorch_translate.tasks.pytorch_translate_task import PytorchTranslateTask
from pytorch_translate.weighted_criterions import (
    WeightedLabelSmoothedCrossEntropyCriterion,
)


@register_task("dual_learning_task")
class DualLearningTask(FairseqTask):
    """A task for training primal model and dual models jointly.
    It takes:
        - unlabelled source (aka source monolingual data for translation task),
        - unlabelled target (aka target monolingual data for translation task),
        - labelled (source, target) (aka parallel data for translation task),
    """

    @staticmethod
    def add_args(parser):
        PytorchTranslateTask.add_args(parser)
        """Add semi-supervised arguments to the parser."""
        parser.add_argument(
            "--dual-criterion",
            default="unsupervised_criterion",
            help="Criterion for jointly train primal and dual models",
        )
        parser.add_argument(
            "--reward-alpha",
            type=float,
            default=0.005,
            help="Hyperparam to weigh two rewards",
        )
        parser.add_argument(
            "--soft-updates",
            type=int,
            metavar="N",
            default=15000,
            help="Number of updates before training with mono",
        )
        parser.add_argument(
            "--train-mono-source-binary-path",
            default="",
            metavar="FILE",
            help="Path for the binary file containing monolingual source "
            "training examples.",
        )
        parser.add_argument(
            "--train-mono-target-binary-path",
            default="",
            metavar="FILE",
            help="Path for the binary file containing monolingual target "
            "training examples.",
        )
        parser.add_argument(
            "--forward-source-vocab-file",
            default="",
            metavar="FILE",
            help="Path to text file representing the dictionary of tokens to use. "
            "If the file does not exist, the dict is auto-generated from source "
            "training data and saved as that file.",
        )
        parser.add_argument(
            "--forward-target-vocab-file",
            default="",
            metavar="FILE",
            help="Path to text file representing the dictionary of tokens to use. "
            "If the file does not exist, the dict is auto-generated from source "
            "training data and saved as that file.",
        )
        parser.add_argument(
            "--forward-train-source-binary-path",
            default="",
            metavar="FILE",
            help="Path for the binary file containing source training "
            "examples for forward model.",
        )
        parser.add_argument(
            "--forward-train-target-binary-path",
            default="",
            metavar="FILE",
            help="Path for the binary file containing target training "
            "examples for forward model.",
        )
        parser.add_argument(
            "--forward-eval-source-binary-path",
            default="",
            metavar="FILE",
            help="Path for the binary file containing source valid "
            "examples for forward model.",
        )
        parser.add_argument(
            "--forward-eval-target-binary-path",
            default="",
            metavar="FILE",
            help="Path for the binary file containing target training "
            "examples for forward model.",
        )
        parser.add_argument(
            "--backward-source-vocab-file",
            default="",
            metavar="FILE",
            help="Path to text file representing the dictionary of tokens to use. "
            "If the file does not exist, the dict is auto-generated from source "
            "training data and saved as that file.",
        )
        parser.add_argument(
            "--backward-target-vocab-file",
            default="",
            metavar="FILE",
            help="Path to text file representing the dictionary of tokens to use. "
            "If the file does not exist, the dict is auto-generated from source "
            "training data and saved as that file.",
        )
        parser.add_argument(
            "--backward-train-source-binary-path",
            default="",
            metavar="FILE",
            help="Path for the binary file containing source training "
            "examples for backward model.",
        )
        parser.add_argument(
            "--backward-train-target-binary-path",
            default="",
            metavar="FILE",
            help="Path for the binary file containing target training "
            "examples for backward model.",
        )
        parser.add_argument(
            "--backward-eval-source-binary-path",
            default="",
            metavar="FILE",
            help="Path for the binary file containing source valid "
            "examples for backward model.",
        )
        parser.add_argument(
            "--backward-eval-target-binary-path",
            default="",
            metavar="FILE",
            help="Path for the binary file containing target training "
            "examples for backwawrd model.",
        )
        parser.add_argument(
            "--remove-eos-at-src", action="store_true", help="If True, remove eos"
        )

    def __init__(
        self, args, primal_src_dict, primal_tgt_dict, dual_src_dict, dual_tgt_dict
    ):
        if not torch.cuda.is_available():
            raise NotImplementedError("Training on CPU is not supported.")
        super().__init__(args)
        self.primal_src_dict = primal_src_dict
        self.primal_tgt_dict = primal_tgt_dict
        self.dual_src_dict = dual_src_dict
        self.dual_tgt_dict = dual_tgt_dict
        self.use_char_source = (args.char_source_vocab_file != "") or (
            args.char_source_vocab_file
        )
        self.task_criterion = UnsupervisedCriterion(args, self).cuda()
        self.criterion = WeightedLabelSmoothedCrossEntropyCriterion(args, self).cuda()
        self.num_update = 0

    def _build_optimizer(self, model):
        if self.args.fp16:
            if torch.cuda.get_device_capability(0)[0] < 7:
                print(
                    "| WARNING: your device does NOT support faster training "
                    "with --fp16, please switch to FP32 which is likely to be"
                    " faster"
                )
            params = list(filter(lambda p: p.requires_grad, model.parameters()))
            self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if torch.cuda.get_device_capability(0)[0] >= 7:
                print("| NOTICE: your device may support faster training with --fp16")
            self._optimizer = optim.build_optimizer(self.args, model.parameters())
        return self._optimizer

    @classmethod
    def setup_task(cls, args, **kwargs):
        cls.source_lang = args.source_lang or "src"
        cls.target_lang = args.target_lang or "tgt"

        primal_src_dict = pytorch_translate_dictionary.Dictionary.load(
            args.source_vocab_file
        )
        primal_tgt_dict = pytorch_translate_dictionary.Dictionary.load(
            args.target_vocab_file
        )
        dual_src_dict = pytorch_translate_dictionary.Dictionary.load(
            args.target_vocab_file
        )
        dual_tgt_dict = pytorch_translate_dictionary.Dictionary.load(
            args.source_vocab_file
        )
        return cls(args, primal_src_dict, primal_tgt_dict, dual_src_dict, dual_tgt_dict)

    def build_criterion(self, args):
        return CRITERION_REGISTRY[args.dual_criterion](args, self)

    def build_model(self, args):
        return self._build_model(args)

    def _build_model(self, args):
        model = dual_learning_models.RNNDualLearningModel.build_model(args, self)
        if not isinstance(model, dual_learning_models.DualLearningModel):
            raise ValueError(
                "DualLearningTask requires a DualLearningModel architecture."
            )
        return model

    def load_dataset(self, split, seed=None):
        """Load split, which is train (monolingual data, optional parallel data),
        or eval (always parallel data).
        """
        if split == self.args.valid_subset:
            # tune set is always parallel
            primal_parallel, _, _ = data_utils.load_parallel_dataset(
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                src_bin_path=self.args.forward_eval_source_binary_path,
                tgt_bin_path=self.args.forward_eval_target_binary_path,
                source_dictionary=self.primal_src_dict,
                target_dictionary=self.primal_tgt_dict,
                split=split,
                remove_eos_from_source=not self.args.append_eos_to_source,
                append_eos_to_target=True,
                char_source_dict=None,
                log_verbose=self.args.log_verbose,
            )
            # now just flip the source and target
            dual_parallel, _, _ = data_utils.load_parallel_dataset(
                source_lang=self.target_lang,
                target_lang=self.source_lang,
                src_bin_path=self.args.backward_eval_source_binary_path,
                tgt_bin_path=self.args.backward_eval_target_binary_path,
                source_dictionary=self.dual_src_dict,
                target_dictionary=self.dual_src_dict,
                split=split,
                remove_eos_from_source=not self.args.append_eos_to_source,
                append_eos_to_target=True,
                char_source_dict=None,
                log_verbose=self.args.log_verbose,
            )
            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict(
                    [
                        ("primal_parallel", primal_parallel),
                        ("dual_parallel", dual_parallel),
                    ]
                )
            )
        elif split == self.args.train_subset:
            src_dataset = data_utils.load_monolingual_dataset(
                self.args.train_mono_source_binary_path, is_source=True
            )
            tgt_dataset = data_utils.load_monolingual_dataset(
                self.args.train_mono_target_binary_path, is_source=True
            )
            primal_source_mono = LanguagePairDataset(
                src=src_dataset,
                src_sizes=src_dataset.sizes,
                src_dict=self.primal_src_dict,
                tgt=None,
                tgt_sizes=None,
                tgt_dict=None,
            )
            dual_source_mono = LanguagePairDataset(
                src=tgt_dataset,
                src_sizes=tgt_dataset.sizes,
                src_dict=self.dual_src_dict,
                tgt=None,
                tgt_sizes=None,
                tgt_dict=None,
            )

            primal_parallel, _, _ = data_utils.load_parallel_dataset(
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                src_bin_path=self.args.forward_train_source_binary_path,
                tgt_bin_path=self.args.forward_train_target_binary_path,
                source_dictionary=self.primal_src_dict,
                target_dictionary=self.primal_tgt_dict,
                split=split,
                remove_eos_from_source=not self.args.append_eos_to_source,
                append_eos_to_target=True,
                char_source_dict=None,
                log_verbose=self.args.log_verbose,
            )
            dual_parallel, _, _ = data_utils.load_parallel_dataset(
                source_lang=self.target_lang,
                target_lang=self.source_lang,
                src_bin_path=self.args.backward_train_source_binary_path,
                tgt_bin_path=self.args.backward_train_target_binary_path,
                source_dictionary=self.dual_src_dict,
                target_dictionary=self.dual_src_dict,
                split=split,
                remove_eos_from_source=not self.args.append_eos_to_source,
                append_eos_to_target=True,
                char_source_dict=None,
                log_verbose=self.args.log_verbose,
            )
            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict(
                    [
                        ("primal_parallel", primal_parallel),
                        ("dual_parallel", dual_parallel),
                        ("primal_source", primal_source_mono),
                        ("dual_source", dual_source_mono),
                    ]
                )
            )
        else:
            raise ValueError("Invalid data split.")

    @property
    def source_dictionary(self):
        return self.primal_src_dict

    @property
    def target_dictionary(self):
        return self.primal_tgt_dict

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None
        return utils.move_to_cuda(sample)

    def _get_src_dict(self, model_key):
        if model_key == "primal":
            return self.primal_src_dict
        else:
            return self.dual_src_dict

    def _get_tgt_dict(self, model_key):
        if model_key == "primal":
            return self.primal_tgt_dict
        else:
            return self.dual_tgt_dict

    def _get_dual(self, model_key):
        if model_key == "primal":
            return "dual"
        else:
            return "primal"

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        # Apply unsupervised dual learning objectives to both types of
        # monolingual data
        self.num_update += 1

        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, {}
        for model_key in model.task_keys:
            data_keys = ["source", "parallel"]
            for data_key in data_keys:
                sample_key = f"{model_key}_{data_key}"
                if sample[sample_key] is None:
                    continue
                if data_key == "parallel":
                    loss, sample_size, logging_output = self.criterion(
                        model.models[model_key], sample[sample_key]
                    )
                    if ignore_grad:
                        loss *= 0
                    optimizer.backward(loss)
                    agg_loss += loss.detach().item()
                    agg_sample_size += sample_size
                    agg_logging_output[sample_key] = logging_output
                if data_key == "source" and self.num_update > self.args.soft_updates:
                    total_loss, sample_size, logging_output = self.task_criterion(
                        sample[sample_key],
                        model.models[model_key],
                        optimizer,
                        self._get_tgt_dict(model_key),
                        model.models[self._get_dual(model_key)],
                        optimizer,
                        self._get_src_dict(model_key),
                        len_penalty=self.args.length_penalty,
                        unk_reward=self.args.unk_reward,
                        word_reward=self.args.word_reward,
                    )
                    agg_loss += total_loss
                    agg_sample_size += sample_size
                    agg_logging_output[sample_key] = logging_output[model_key]

        return agg_loss, agg_sample_size, agg_logging_output

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        parallel_keys = [
            "primal_parallel",
            "dual_parallel",
            "primal_source",
            "dual_source",
        ]
        agg_logging_outputs = {}
        for lang_pair in parallel_keys:
            agg_output = []
            for logging_output in logging_outputs:
                if lang_pair in logging_output.keys():
                    agg_output.append(logging_output[lang_pair])
            if len(agg_output) > 0:
                agg_logging_outputs[
                    lang_pair
                ] = criterion.__class__.aggregate_logging_outputs(agg_output)

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

    def valid_step(self, sample, model, criterion):
        agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, {}
        if sample["primal_parallel"] is not None:
            model.eval()
            with torch.no_grad():
                loss, sample_size, logging_output = self.criterion(
                    model.models["primal"], sample["primal_parallel"]
                )
                agg_loss += loss.data.item()
                agg_sample_size += sample_size
                agg_logging_output["primal_parallel"] = logging_output
        if sample["dual_parallel"] is not None:
            model.eval()
            with torch.no_grad():
                loss, sample_size, logging_output = self.criterion(
                    model.models["dual"], sample["dual_parallel"]
                )
                agg_loss += loss.data.item()
                agg_sample_size += sample_size
                agg_logging_output["dual_parallel"] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output
