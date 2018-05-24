#!/usr/bin/env python3

import argparse
import os
import tempfile

from fairseq import indexed_dataset, tokenizer
from pytorch_translate import dictionary as pytorch_translate_dictionary
from pytorch_translate import utils


def add_args(parser):
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
        "--train-source-binary-prefix",
        default="",
        help="Path prefix for the output binary and index files containing "
        "source training examples.",
    )
    group.add_argument(
        "--train-target-binary-prefix",
        default="",
        help="Path prefix for the output binary and index files containing "
        "target training examples.",
    )
    group.add_argument(
        "--eval-source-binary-prefix",
        default="",
        help="Path prefix for the output binary and index files containing "
        "source eval examples for calculating validation loss and BLEU scores.",
    )
    group.add_argument(
        "--eval-target-binary-prefix",
        default="",
        help="Path prefix for the output binary and index files containing "
        "target eval examples for calculating validation loss and BLEU scores.",
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


def validate_args(args):
    if not (
        (args.train_source_text_file or args.train_source_binary_prefix)
        and (args.train_target_text_file or args.train_target_binary_prefix)
        and (args.eval_source_text_file or args.eval_source_binary_prefix)
        and (args.eval_target_text_file or args.eval_target_binary_prefix)
    ):
        raise ValueError(
            "At least one of --*_text_file or --*_binary_prefix flags must be "
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


def binarize_text_file(
    text_file: str,
    dictionary: pytorch_translate_dictionary.Dictionary,
    output_prefix: str,
    append_eos: bool,
    reverse_order: bool,
) -> str:
    if not output_prefix:
        fd, output_prefix = tempfile.mkstemp()
        # We only need a unique file name prefix, since the helper functions
        # take care of actually creating the file.
        os.close(fd)

    print(
        f"Outputting binarized version of {text_file} to "
        f"{indexed_dataset.data_file_path(output_prefix)} and "
        f"{indexed_dataset.index_file_path(output_prefix)}"
    )
    builder = indexed_dataset.IndexedDatasetBuilder(
        indexed_dataset.data_file_path(output_prefix)
    )

    def consumer(tensor):
        builder.add_item(tensor)

    counters = tokenizer.Tokenizer.binarize(
        filename=text_file,
        dict=dictionary,
        consumer=consumer,
        append_eos=append_eos,
        reverse_order=reverse_order,
    )
    print(
        f"Binarizing {text_file}: {counters['nseq']} sents, "
        f"{counters['ntok']} tokens, "
        f"{100 * counters['nunk'] / counters['ntok']:.3}% replaced by "
        f"{dictionary.unk_word}."
    )

    builder.finalize(indexed_dataset.index_file_path(output_prefix))

    return output_prefix


def preprocess_corpora(args):
    validate_args(args)

    # Additional text preprocessing options could be added here before
    # binarizing.

    source_dict = pytorch_translate_dictionary.Dictionary.build_vocab_file_if_nonexistent(
        corpus_file=args.train_source_text_file,
        vocab_file=args.source_vocab_file,
        max_vocab_size=args.source_max_vocab_size,
        tokens_with_penalty=None,
    )
    if args.char_source_vocab_file:
        # TODO: assign results and pass to new and improved binarize_text_file
        pytorch_translate_dictionary.Dictionary.build_vocab_file_if_nonexistent(
            corpus_file=args.train_source_text_file,
            vocab_file=args.char_source_vocab_file,
            max_vocab_size=args.char_source_max_vocab_size,
            tokens_with_penalty=None,
            is_char_vocab=True,
        )
    if args.train_source_text_file:
        args.train_source_binary_prefix = binarize_text_file(
            text_file=args.train_source_text_file,
            dictionary=source_dict,
            output_prefix=args.train_source_binary_prefix,
            append_eos=args.append_eos_to_source,
            reverse_order=args.reverse_source,
        )
    if args.eval_source_text_file:
        args.eval_source_binary_prefix = binarize_text_file(
            text_file=args.eval_source_text_file,
            dictionary=source_dict,
            output_prefix=args.eval_source_binary_prefix,
            append_eos=args.append_eos_to_source,
            reverse_order=args.reverse_source,
        )

    target_dict = pytorch_translate_dictionary.Dictionary.build_vocab_file_if_nonexistent(
        corpus_file=args.train_target_text_file,
        vocab_file=args.target_vocab_file,
        max_vocab_size=args.target_max_vocab_size,
        tokens_with_penalty=args.penalized_target_tokens_file,
    )
    # For target sentences, we always append EOS tokens, and never reverse
    # their order.
    if args.train_target_text_file:
        args.train_target_binary_prefix = binarize_text_file(
            text_file=args.train_target_text_file,
            dictionary=target_dict,
            output_prefix=args.train_target_binary_prefix,
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
    if args.eval_target_text_file:
        args.eval_target_binary_prefix = binarize_text_file(
            text_file=args.eval_target_text_file,
            dictionary=target_dict,
            output_prefix=args.eval_target_binary_prefix,
            append_eos=True,
            reverse_order=False,
        )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Translate - preprocessing")
    add_args(parser)
    args = parser.parse_args()
    preprocess_corpora(args)


if __name__ == "__main__":
    main()
