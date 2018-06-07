#!/usr/bin/env python3

from pytorch_translate import utils


def add_dataset_args(parser, train=False, gen=False):
    """Same as fairseq.options.add_dataset_args but without
    the "data" argument"""
    group = parser.add_argument_group("Dataset and data loading")
    group.add_argument("data", metavar="DIR", nargs="?",
                       help="path to data directory. "
                       "This is not needed but kept for backward compatibility")
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
    return group


def expand_generation_args(group, train=False, gen=False):
    """Expands the generation related arguments with pytorch_translate
    specific arguments"""
    group.add_argument(
        "--word-reward",
        type=float,
        default=0.0,
        help=(
            "Value to add to (log-prob) score for each token except EOS. "
            "IMPORTANT NOTE: higher values of --lenpen and --word-reward "
            "both encourage longer translations, while higher values of "
            "--unkpen penalize UNKs more."
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
    # Add filenames arguments for generation
    if gen:
        group.add_argument(
            "--source-vocab-file",
            default="",
            metavar="FILE",
            help="Path to text file representing the Dictionary to use.",
        )
        group.add_argument(
            "--target-vocab-file",
            default="",
            metavar="FILE",
            help="Path to text file representing the Dictionary to use.",
        )
        group.add_argument(
            "--source-text-file",
            default="",
            metavar="FILE",
            help="Path to raw text file containing examples in source dialect. "
            "This overrides what would be loaded from the data dir.",
        )
        group.add_argument(
            "--target-text-file",
            default="",
            metavar="FILE",
            help="Path to raw text file containing examples in target dialect. "
            "This overrides what would be loaded from the data dir.",
        )
        group.add_argument(
            "--translation-output-file",
            default="",
            type=str,
            metavar="FILE",
            help="Path to text file to store the output of the model. ",
        )
    return group
