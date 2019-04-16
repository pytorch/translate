#!/usr/bin/env python3

import os

import torch
from pytorch_translate import constants, utils


def check_unsupported_fairseq_flags(args):
    UNSUPPORTED_FAIRSEQ_FLAGS = [
        # Use --save-interval-updates instead.
        ("save_interval", 1),
        # Look at --num-avg-checkpoints and --auto-clear-checkpoints.
        ("keep_interval_updates", -1),
        # We always save checkpoints at the end of an epoch.
        ("no_epoch_checkpoints", False),
        # We always save checkpoints.
        ("no_save", False),
        # We run validation every time we save a checkpoint, so this is effectively
        # controlled by --save-interval-updates as well.
        ("validate_interval", 1),
    ]

    for (flag_name, default_value) in UNSUPPORTED_FAIRSEQ_FLAGS:
        if hasattr(args, flag_name):
            if getattr(args, flag_name) != default_value:
                raise ValueError(
                    f"Found unsupported Fairseq flag "
                    f"--{flag_name.replace('_', '-')} with non-default value "
                    f"of {getattr(args, flag_name)} "
                )
            setattr(args, flag_name, None)


def add_dataset_args(parser, train=False, gen=False):
    """Same as fairseq.options.add_dataset_args but without
    the "data" argument"""
    group = parser.add_argument_group("Dataset and data loading")
    group.add_argument(
        "data",
        metavar="DIR",
        nargs="?",
        help="path to data directory. "
        "This is not needed but kept for backward compatibility",
    )
    group.add_argument(
        "--num-workers",
        default=0,
        type=int,
        metavar="N",
        help="how many subprocesses to use for data loading",
    )
    group.add_argument(
        "--skip-invalid-size-inputs-valid-test",
        action="store_true",
        help="Ignore too long or too short lines in valid and test set",
    )
    group.add_argument(
        "--max-tokens",
        default=5000,
        type=int,
        metavar="N",
        help="maximum number of tokens in a batch",
    )
    group.add_argument(
        "--max-sentences",
        "--batch-size",
        type=int,
        metavar="N",
        help="maximum number of sentences in a batch",
    )
    if train:
        group.add_argument(
            "--train-subset",
            default="train",
            metavar="SPLIT",
            choices=["train", "valid", "test"],
            help="data subset to use for training (train, valid, test)",
        )
        group.add_argument(
            "--valid-subset",
            default="valid",
            metavar="SPLIT",
            help="comma separated list of data subsets to use"
            " for validation (train, valid, valid1,test, test1)",
        )
        group.add_argument(
            "--max-sentences-valid",
            type=int,
            metavar="N",
            help="maximum number of sentences in a validation batch"
            " (defaults to --max-sentences)",
        )
    if gen:
        group.add_argument(
            "--gen-subset",
            default="test",
            metavar="SPLIT",
            help="data subset to generate (train, valid, test)",
        )
        group.add_argument(
            "--num-shards",
            default=1,
            type=int,
            metavar="N",
            help="shard generation over N shards",
        )
        group.add_argument(
            "--shard-id",
            default=0,
            type=int,
            metavar="ID",
            help="id of the shard to generate (id < num_shards)",
        )
    return group


def add_preprocessing_args(parser):
    # Args related to dataset.
    group = parser.add_argument_group("Preprocess data")
    group.add_argument(
        "--source-vocab-file",
        default="",
        metavar="FILE",
        help="Path to text file representing the dictionary of tokens to use. "
        "If the file does not exist, the dict is auto-generated from source "
        "training data and saved as that file.",
    )
    group.add_argument(
        "--source-max-vocab-size",
        default=-1,
        type=int,
        metavar="N",
        help="If a new vocab file needs to be generated, restrict it to the "
        "top N most common words. If we re-use an existing vocab file, this "
        "flag will have no effect. A value of < 0 means no max size.",
    )
    group.add_argument(
        "--source-bpe-cont-marker",
        default=None,
        type=str,
        metavar="CONT",
        help="Source BPE continuation marker. You should only specify this if "
        "you are using a BPE source vocab that has an continuation marker "
        "suffix. Note that this is the default BPE format in fairseq. Ex: '@@'",
    )
    group.add_argument(
        "--source-bpe-end-marker",
        default=None,
        type=str,
        metavar="END",
        help="Source BPE end marker. You should only specify this if you are "
        "using a BPE source vocab that has an end marker suffix. Ex: '_EOW'",
    )
    group.add_argument(
        "--char-source-vocab-file",
        default="",
        metavar="FILE",
        help="Same as --source-vocab-file except using characters.",
    )
    group.add_argument(
        "--embed-bytes",
        type=utils.bool_flag,
        nargs="?",
        const=True,
        default=False,
        help="If specified along with a character model and set to True, "
        "then we embed bytes instead of characters.",
    )
    group.add_argument(
        "--char-source-max-vocab-size",
        default=-1,
        type=int,
        metavar="N",
        help="Same as --source-max-vocab-size except using characters.",
    )
    group.add_argument(
        "--target-vocab-file",
        default="",
        metavar="FILE",
        help="Path to text file representing the fairseq Dictionary to use. "
        "If the file does not exist, the dict is auto-generated from target "
        "training data and saved as that file.",
    )
    group.add_argument(
        "--target-max-vocab-size",
        default=-1,
        type=int,
        metavar="N",
        help="If a new vocab file needs to be generated, restrict it to the "
        "top N most common words. If we re-use an existing vocab file, this "
        "flag will have no effect. A value of < 0 means no max size.",
    )
    group.add_argument(
        "--target-bpe-cont-marker",
        default=None,
        type=str,
        metavar="CONT",
        help="Target BPE continuation marker. You should only specify this if "
        "you are using a BPE target vocab that has an continuation marker "
        "suffix. Note that this is the default BPE format in fairseq. Ex: '@@'",
    )
    group.add_argument(
        "--target-bpe-end-marker",
        default=None,
        type=str,
        metavar="END",
        help="Target BPE end marker. You should only specify this if you are "
        "using a BPE target vocab that has an end marker suffix. Ex: '_EOW'",
    )
    group.add_argument(
        "--train-source-text-file",
        default="",
        metavar="FILE",
        help="Path to text file containing source training examples.",
    )
    group.add_argument(
        "--train-target-text-file",
        default="",
        metavar="FILE",
        help="Path to text file containing target training examples.",
    )
    group.add_argument(
        "--eval-source-text-file",
        default="",
        metavar="FILE",
        help="Path to text file containing source eval examples for "
        "calculating validation loss and BLEU eval scores.",
    )
    group.add_argument(
        "--eval-target-text-file",
        default="",
        metavar="FILE",
        help="Path to text file containing target eval examples for "
        "calculating validation loss and BLEU eval scores.",
    )
    group.add_argument(
        "--train-source-binary-path",
        default="",
        help="Path for the binary file containing source training examples.",
    )
    group.add_argument(
        "--train-target-binary-path",
        default="",
        help="Path for the binary file containing target training examples.",
    )
    group.add_argument(
        "--train-weights-path",
        default="",
        metavar="FILE",
        help="Path to text file of weight (0 to 1) for each train example.."
        "If left empty, all examples will receive equal weights.",
    )
    group.add_argument(
        "--eval-source-binary-path",
        default="",
        help="Path for the binary file containing source eval examples for "
        "calculating validation loss and BLEU scores.",
    )
    group.add_argument(
        "--eval-target-binary-path",
        default="",
        help="Path for the binary file containing target eval examples for "
        "calculating validation loss and BLEU scores.",
    )

    group.add_argument(
        "--multiling-encoder-lang",
        action="append",
        metavar="SRC",
        help="For multilingual models only. Use this argument repeatedly to "
        "specify a list of encoder languages. The multilingual model contains "
        "a separate encoder for each language in this list.",
    )
    group.add_argument(
        "--multiling-decoder-lang",
        action="append",
        metavar="TARGET",
        help="For multilingual models only. Use this argument repeatedly to "
        "specify a list of decoder languages. The multilingual model contains "
        "a separate decoder for each language in this list.",
    )
    group.add_argument(
        "--multiling-source-lang",
        action="append",
        metavar="SRC",
        help="For multilingual models only. Use this argument repeatedly to "
        "specify a list of corpus source languages, where the n-th language is "
        "the source language of the n-th training corpus. Each entry must be "
        "in --multiling-encoder-lang.",
    )
    group.add_argument(
        "--multiling-target-lang",
        action="append",
        metavar="TARGET",
        help="For multilingual models only. Use this argument repeatedly to "
        "specify a list of corpus target languages, where the n-th language is "
        "the target language of the n-th training corpus. Each entry must be "
        "in --multiling-decoder-lang.",
    )
    group.add_argument(
        "--multiling-source-vocab-file",
        action="append",
        metavar="FILE",
        help="For multilingual models only. Use this argument repeatedly to "
        "specify the path to the dictionary for the n-th entry in "
        "--multiling-encoder-lang",
    )
    group.add_argument(
        "--multiling-target-vocab-file",
        action="append",
        metavar="FILE",
        help="For multilingual models only. Use this argument repeatedly to "
        "specify the path to the dictionary for the n-th entry in "
        "--multiling-decoder-lang",
    )
    group.add_argument(
        "--multiling-train-source-text-file",
        action="append",
        metavar="FILE",
        help="For multilingual models only. Use this argument repeatedly to "
        "specify paths to training source samples. The n-th entry should be "
        "in the n-th language in --multiling-source-lang.",
    )
    group.add_argument(
        "--multiling-train-target-text-file",
        action="append",
        metavar="FILE",
        help="For multilingual models only. Use this argument repeatedly to "
        "specify paths to training target samples. The n-th entry should be "
        "in the n-th language in --multiling-target-lang.",
    )
    group.add_argument(
        "--multiling-train-oversampling",
        action="append",
        type=int,
        help="For multilingual models only. Use this argument repeatedly to "
        "oversample corpora. The n-th training corpus is oversampled by the n-"
        "the entry. No oversampling if not specified.",
    )
    group.add_argument(
        "--multiling-eval-source-text-file",
        action="append",
        metavar="FILE",
        help="For multilingual models only. Use this argument repeatedly to "
        "specify paths to eval source samples. The n-th entry should be "
        "in the n-th language in --multiling-source-lang.",
    )
    group.add_argument(
        "--multiling-eval-target-text-file",
        action="append",
        metavar="FILE",
        help="For multilingual models only. Use this argument repeatedly to "
        "specify paths to eval target samples. The n-th entry should be "
        "in the n-th language in --multiling-target-lang.",
    )
    group.add_argument(
        "--multiling-rescale-grads",
        type=utils.bool_flag,
        nargs="?",
        const=True,
        default=False,
        help=(
            "If true, rescale gradients based on the number of training "
            "samples a specific component has received in a training batch."
        ),
    )

    group.add_argument(
        "--penalized-target-tokens-file",
        default="",
        metavar="FILE",
        help="Path to text file of tokens to receive a penalty in decoding."
        "If left empty, no penalty will be applied",
    )

    group.add_argument(
        "--append-eos-to-source",
        type=utils.bool_flag,
        nargs="?",
        const=True,
        default=False,
        help=("If true, append EOS to source sentences (instead of just target)."),
    )
    group.add_argument(
        "--reverse-source",
        type=utils.bool_flag,
        nargs="?",
        const=True,
        default=True,
        help=("If true, feed source sentence to model in reverse order."),
    )


def validate_preprocessing_args(args):
    if (
        not (
            (
                getattr(args, "train_source_text_file", None)
                or getattr(args, "train_source_binary_path", None)
                or getattr(args, "multiling_train_source_text_file", None)
            )
            and (
                getattr(args, "train_target_text_file", None)
                or getattr(args, "train_target_binary_path", None)
                or getattr(args, "multiling_train_target_text_file", None)
            )
            and (
                getattr(args, "eval_source_text_file", None)
                or getattr(args, "eval_source_binary_path", None)
                or getattr(args, "multiling_eval_source_text_file", None)
            )
            and (
                getattr(args, "eval_target_text_file", None)
                or getattr(args, "eval_target_binary_path", None)
                or getattr(args, "multiling_eval_target_text_file", None)
            )
        )
        and args.task != "dual_learning_task"
    ):
        raise ValueError(
            "At least one of --*_text_file or --*_binary_path flags must be "
            "specified for each of --{train, eval}_{source, target}_*"
        )

    for file_type in (
        "train_source_text_file",
        "train_target_text_file",
        "eval_source_text_file",
        "eval_target_text_file",
    ):
        text_file = getattr(args, file_type)
        if text_file and not os.path.isfile(text_file):
            raise ValueError(
                f"Please specify an existing text file for "
                f"--{file_type}={text_file}"
            )

    for file_type in ("source_vocab_file", "target_vocab_file"):
        vocab_file = getattr(args, file_type)
        if not vocab_file:
            raise ValueError(
                f"--{file_type} must be specified - even if you don't have "
                f"a vocab file, you must still specify a location "
                f"for it to be written to."
            )

    if args.task == constants.SEMI_SUPERVISED_TASK and not (
        getattr(args, "train_mono_source_binary_path", None)
        or getattr(args, "train_mono_target_binary_path", None)
        or getattr(args, "train_mono_source_text_file", None)
    ):
        raise ValueError(
            "For semisupervised training, at least one of --*_text_file or "
            "--*_binary_path flags must be specified for at least one of "
            "--train_mono_{source, target}_*"
        )

        for file_type in ("train_mono_source_text_file", "train_mono_target_text_file"):
            file_path = getattr(args, file_type)
            if file_path and not os.path.isfile(file_path):
                raise ValueError(
                    f"Please specify an existing text file for --{file_type}="
                    f"{file_path}"
                )


def expand_optimization_args(group):
    """Expands the optimization related arguments with pytorch_translate
    specific arguments"""
    group.add_argument(
        "--local-num-gpus",
        default=torch.cuda.device_count(),
        type=int,
        metavar="N",
        help=(
            "The number of local GPUs to use for training on this machine. "
            "Defaults to using all visible GPUs. This should be "
            "<= --distributed-world-size."
        ),
    )
    group.add_argument(
        "--stop-time-hr",
        default=-1.0,
        type=float,
        metavar="N",
        help="Stops training after N hours have elapsed. Use decimal values "
        "for sub-hourly granularity. A value of < 0 disables this.",
    )
    group.add_argument(
        "--stop-no-best-validate-loss",
        default=-1,
        type=int,
        metavar="N",
        help="Stops training after N validations have been run without "
        "achieving a better loss than before. Note that this is affected by "
        "--save-interval-updates in how frequently we run validation in the "
        "first place. A value of < 0 disables this.",
    )
    group.add_argument(
        "--stop-no-best-bleu-eval",
        default=-1,
        type=int,
        metavar="N",
        help="Stops training after N evals have been run without "
        "achieving a better BLEU score than before. Note that this is affected "
        "by --save-interval-updates in how frequently we run BLEU eval "
        "in the first place. A value of < 0 disables this.",
    )
    group.add_argument(
        "--shrink-lr-no-best-bleu-eval",
        default=5,
        type=int,
        metavar="N",
        help="Decay learning rate after N evals have been run without "
        "achieving a better BLEU score than before. This is to achieve "
        "decay lr within an epoch, independent of lr_scheduler. "
        "Note that this is affected by --save-interval-updates in "
        "how frequently we run BLEU eval in the first place. "
        "A value of < 0 disables this.",
    )
    group.add_argument(
        "--pruning-percentile",
        type=int,
        default=0,
        help="Proportion of weights to prune. A value <=0 disables pruning."
        " By default, prunes weights uniformly and ignores bias terms.",
    )
    group.add_argument(
        "--parameters-to-prune",
        default="all",
        help="Names of layers to prune. Layers are pruned if the argument is "
        "a substring of the layer name. Options are 'all', 'embed', 'lstm'. ",
    )
    group.add_argument(
        "--loss-beam",
        type=int,
        default=0,
        help="Beam size to use for 'sequence_nll' loss and 'sequence_risk' "
        "loss. If zero, use --beam.",
    )
    group.add_argument(
        "--disable-eval-bleu",
        nargs="?",
        const=True,
        default=False,
        type=utils.bool_flag,
        help=("disable bleu score evaluation on tune dataset"),
    )

    return group


def expand_checkpointing_args(group):
    """Expands the checkpointing related arguments with pytorch_translate
    specific arguments"""
    group.add_argument(
        "--auto-clear-checkpoints",
        type=utils.bool_flag,
        nargs="?",
        const=True,
        default=True,
        help=(
            "If True, we keep only the last --num-avg-checkpoints checkpoints "
            "on disk and delete all older checkpoints."
        ),
    )
    group.add_argument(
        "--num-avg-checkpoints",
        default=1,
        type=int,
        metavar="N",
        help=(
            "Average over the last N checkpoints when saving "
            "averaged checkpoints and when doing BLEU eval. Must be >=1."
        ),
    )
    group.add_argument(
        "--pretrained-checkpoint-file",
        default="",
        type=str,
        metavar="FILE",
        help=(
            "Allows the user to resume or fine-tune training (possibly with "
            "different parameter) from another run's checkpoint. Note that "
            "if there's already an existing checkpoint at "
            "os.path.join(--save-dir, --restore-file), this "
            "flag will have no effect. We prefer using --save-dir over this "
            "flag to ensure that if user resumes training on a model that "
            "was originaly initialized from another run's checkpoint, we won't "
            "wipe out all progress in --save-dir by re-loading the previous "
            "run's checkpoint."
        ),
    )
    group.add_argument(
        "--load-pretrained-checkpoint-state",
        type=utils.bool_flag,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Whether to also load optimizer extra_state (epoch, "
            "training time, etc) when loading from --pretrained-checkpoint-file. "
            "If false, only model weights are loaded from that file. "
            "Note that this has no effect when restoring from --restore-file."
        ),
    )

    return group


def expand_generation_args(group, train=False):
    """Expands the generation related arguments with pytorch_translate
    specific arguments"""
    group.add_argument(
        "--word-reward",
        type=float,
        default=0.0,
        help=(
            "Value to add to (log-prob) score for each token except EOS. "
            "Value < 0 encourages shorter translations, while > 0 "
            "(the usual case) encourages longer translations "
            "(similar to --length-penalty)."
        ),
    )
    group.add_argument(
        "--unk-reward",
        type=float,
        default=0.0,
        help=(
            "Value to add to (log-prob) score for UNK tokens. "
            "Value < 0 (the usual case) encourages fewer UNKs, while > 0 "
            "encourages more UNKs."
        ),
    )
    group.add_argument(
        "--length-penalty",
        type=float,
        default=0.0,
        help=(
            "When >0 scores are normalized according to length (divided by "
            "length^length_penalty). Effectively overrides word_reward when"
            "in use. NOTE: supersedes --lenpen."
        ),
    )
    group.add_argument(
        "--model-weights",
        default="",
        help=(
            "Interpolation weights for ensembles. Comma-separated list of "
            "floats with length equal to the number of models in the ensemble."
        ),
    )
    group.add_argument(
        "--report-oracle-bleu",
        type=utils.bool_flag,
        nargs="?",
        const=True,
        default=False,
        help=(
            "During evaluation, determine best among top-k outputs (where k "
            "is controlled by --nbest) for each sentence by smoothed "
            "sentence-level BLEU and report overall BLEU score for these "
            "sentences."
        ),
    )
    group.add_argument(
        "--output-hypos-binary-path",
        default=None,
        type=str,
        help=(
            "Optional filename to save output hypotheses (binary format "
            "and EOS-terminated, suitable for use as training targets)"
        ),
    )
    group.add_argument(
        "--translation-info-export-path",
        default=None,
        type=str,
        help=("Optional path to save translation info output in pickled format"),
    )
    group.add_argument(
        "--diversity-sibling-gamma",
        type=float,
        default=0.0,
        help=("The diversity rate of sibling_rank for generating diverse beams"),
    )

    # These arguments are only used during training
    if train:
        group.add_argument(
            "--multi-model-restore-files",
            default=None,
            type=str,
            nargs="+",
            help=(
                "If --multi-encoder = --multi-decoder > 1, this option makes "
                "it possible to initialize individual model weights from "
                "existing checkpoints of separate training runs."
            ),
        )
    return group


def validate_generation_args(args):
    assert args.unkpen == 0, (
        "PyTorch Translate does not use fairseq's --unkpen flag. "
        "Use --unk-reward instead, and check the flag description regarding "
        "sign polarity meaning."
    )
    assert args.lenpen == 1, (
        "Argument --lenpen is IGNORED by pytorch_translate. Use "
        "--length-penalty instead."
    )
    if "num_avg_checkpoints" in args:
        assert args.num_avg_checkpoints >= 1, "--num-avg-checkpoints must be >= 1."


def add_verbosity_args(parser, train=False):
    verbosity_group = parser.add_argument_group("Verbosity")
    verbosity_group.add_argument(
        "--log-verbose",
        action="store_true",
        help="Whether to output more verbose logs for debugging/profiling.",
    )
    verbosity_group.add_argument(
        "--args-verbosity",
        default=1,
        type=int,
        choices=[0, 1, 2],
        help="Level of verbosity when printing the arguments (0: don't print "
        "the arguments; 1: print the Namespace object; 2: print all the "
        "arguments, one per line). The default is 1."
        "one per line)",
    )
    return verbosity_group


def print_args(args):
    args_verbosity = getattr(args, "args_verbosity", 1)
    if args_verbosity == 2:
        args_sorted = sorted(vars(args).items())
        for name, value in args_sorted:
            print(f"{name}={value}")
    elif args_verbosity == 1:
        print(args)
    elif args_verbosity == 0:
        return
    else:
        raise ValueError("Please specify an argument verbosity level between 0 and 2")
