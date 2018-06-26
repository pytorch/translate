#!/usr/bin/env python3

import os
from pytorch_translate import utils


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
        "-s", "--source-lang", default="src", metavar="SRC", help="source language"
    )
    group.add_argument(
        "-t", "--target-lang", default="tgt", metavar="TARGET", help="target language"
    )
    group.add_argument(
        "--max-source-positions",
        default=1024,
        type=int,
        metavar="N",
        help="max number of tokens in the source sequence",
    )
    group.add_argument(
        "--max-target-positions",
        default=1024,
        type=int,
        metavar="N",
        help="max number of tokens in the target sequence",
    )
    group.add_argument(
        "--skip-invalid-size-inputs-valid-test",
        action="store_true",
        help="Ignore too long or too short lines in valid and test set",
    )
    group.add_argument(
        "--max-tokens",
        default=6000,
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
        "--char-source-vocab-file",
        default="",
        metavar="FILE",
        help="Same as --source-vocab-file except using characters.",
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
    if not (
        (
            args.train_source_text_file
            or args.train_source_binary_path
            or args.multiling_train_source_text_file
        )
        and (
            args.train_target_text_file
            or args.train_target_binary_path
            or args.multiling_train_target_text_file
        )
        and (
            args.eval_source_text_file
            or args.eval_source_binary_path
            or args.multiling_eval_source_text_file
        )
        and (
            args.eval_target_text_file
            or args.eval_target_binary_path
            or args.multiling_eval_target_text_file
        )
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
        file = getattr(args, file_type)
        if file and not os.path.isfile(file):
            raise ValueError(
                f"Please specify an existing text file for --{file_type}={file}"
            )

    for file_type in ("source_vocab_file", "target_vocab_file"):
        file = getattr(args, file_type)
        if not file:
            raise ValueError(
                f"--{file_type} must be specified - even if you don't have "
                f"a vocab file, you must still specify a location "
                f"for it to be written to."
            )


def expand_optimization_args(group):
    """Expands the optimization related arguments with pytorch_translate
    specific arguments"""
    group.add_argument(
        "--subepoch-validate-interval",
        default=0,
        type=int,
        metavar="N",
        help="Calculates loss over the validation set every N batch updates. "
        "Note that validation is done at the end of every epoch regardless. "
        "A value of <= 0 disables this.",
    )
    group.add_argument(
        "--stop-time-hr",
        default=-1,
        type=int,
        metavar="N",
        help="Stops training after N hours have elapsed. "
        "A value of < 0 disables this.",
    )
    group.add_argument(
        "--stop-no-best-validate-loss",
        default=-1,
        type=int,
        metavar="N",
        help="Stops training after N validations have been run without "
        "achieving a better loss than before. Note that this is affected by "
        "--validation-interval in how frequently we run validation in the "
        "first place. A value of < 0 disables this.",
    )
    group.add_argument(
        "--stop-no-best-bleu-eval",
        default=-1,
        type=int,
        metavar="N",
        help="Stops training after N evals have been run without "
        "achieving a better BLEU score than before. Note that this is affected "
        "by --generate-bleu-eval-interval in how frequently we run BLEU eval "
        "in the first place. A value of < 0 disables this.",
    )
    group.add_argument(
        "--pruning-percentile",
        type=int,
        default=0,
        help="Proportion of weights to prune after initially training to either "
        "BLEU score stagnation or time limit. A value <=0 disables pruning.",
    )
    group.add_argument(
        "--retrain-lr-ratio",
        type=float,
        default=0.0,
        help="Learning rate for retraining the model after pruning is performed "
        "relative to learning rate before pruning. Affected by lr scheduler. "
        "Retraining is not performed if --pruning-percentile is set to disable "
        "pruning.",
    )

    return group


def expand_checkpointing_args(group):
    """Expands the checkpointing related arguments with pytorch_translate
    specific arguments"""
    group.add_argument(
        "--no-end-of-epoch-checkpoints",
        action="store_true",
        help="Disables saving checkpoints at the end of the epoch. "
        "This differs from --no-save and --no-epoch-checkpoints in that it "
        "still allows for intra-epoch checkpoints if --save-interval is set.",
    )
    group.add_argument(
        "--max-checkpoints-kept",
        default=-1,
        type=int,
        metavar="N",
        help="Keep at most the last N checkpoints file around. "
        "A value < -1 keeps all. "
        "When --generate-bleu-eval-avg-checkpoints is used and is > N, the "
        "number of checkpoints kept around is automatically adjusted "
        "to allow BLEU to work properly.",
    )
    group.add_argument(
        "--pretrained-weights-file",
        default="",
        type=str,
        help="Load the model weights from this file. This is different from "
        "--restore-file in that it only loads the weights (and not the optimizer "
        "state). This option is superseded by --restore-file so don't use both.",
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
    # These arguments are only used during training
    if train:
        group.add_argument(
            "--generate-bleu-eval-per-epoch",
            action="store_true",
            help="Whether to generate BLEU score eval after each epoch.",
        )
        group.add_argument(
            "--generate-bleu-eval-interval",
            default=0,
            type=int,
            metavar="N",
            help="Does BLEU eval every N batch updates. Note that "
            "--save-interval also affects this - we can only eval as "
            "frequently as a checkpoint is written. A value of <= 0 "
            "disables this.",
        )
        group.add_argument(
            "--generate-bleu-eval-avg-checkpoints",
            default=1,
            type=int,
            metavar="N",
            help="Maximum number of last N checkpoints to average over when "
            "doing BLEU eval. Must be >= 1.",
        )
        group.add_argument(
            "--continuous-averaging-after-epochs",
            type=int,
            default=-1,
            help=(
                "Average parameter values after each step since previous "
                "checkpoint, beginning after the specified number of epochs. "
            ),
        )
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
    pass


def add_verbosity_args(parser, train=False):
    verbosity_group = parser.add_argument_group("Verbosity")

    if train:
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
