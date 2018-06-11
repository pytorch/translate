#!/usr/bin/env python3

import os
import random
import torch

from fairseq import bleu, data, options, progress_bar, tokenizer, utils
from fairseq.meters import TimeMeter
from pytorch_translate import beam_decode
from pytorch_translate import generate as pytorch_translate_generate
from pytorch_translate import data as pytorch_translate_data
from pytorch_translate import dictionary as pytorch_translate_dictionary
from pytorch_translate import utils as pytorch_translate_utils
from pytorch_translate import rnn  # noqa


def get_parser_with_args():
    parser = options.get_parser("Generation")
    options.add_dataset_args(parser, gen=True)
    options.add_generation_args(parser)
    pytorch_translate_generate.add_args(parser)

    group = parser.add_argument_group("Generation")
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

    # Add args related to benchmarking.
    group = parser.add_argument_group("Benchmarking")
    group.add_argument(
        "--increment",
        default=5,
        type=int,
        help="Difference in lengths between synthesized sentences. "
        "Must be integer >=1."
    )
    group.add_argument(
        "--max-length",
        default=100,
        type=int,
        help="Maximum allowed length for synthesized sentences. "
        "Should be greater than --increment."
    )
    group.add_argument(
        "--samples-per-length",
        default=1,
        type=int,
        help="Number of sentences to be synthesized at each length. "
    )

    return parser


def main():
    parser = get_parser_with_args()
    args = parser.parse_args()
    # Disable printout of all source and target sentences
    args.quiet = True
    generate(args)


def assert_test_corpus_and_vocab_files_specified(args):
    assert not args.data, (
        "Specifying a data directory is disabled in FBTranslate since the "
        "fairseq data class is not supported. Please specify "
        "--source-vocab-file, --target-vocab-file"
    )
    assert (
        args.source_vocab_file and os.path.isfile(args.source_vocab_file)
    ), "Please specify a valid file for --source-vocab-file"
    assert (
        args.target_vocab_file and os.path.isfile(args.target_vocab_file)
    ), "Please specify a valid file for --target-vocab_file"


def generate_synthetic_text(dialect, dialect_symbols, args):
    assert args.max_length >= 1, "Please specify a valid maximum length"
    temp_file_name = f'benchmark_text_file_{dialect}'
    with open(temp_file_name, 'w') as temp_file:
        # Short sentence to prime GPU
        temp_file.write(dialect_symbols[0] + "\n")

        for _ in range(args.samples_per_length):
            for sentence_length in range(
                args.increment, args.max_length, args.increment
            ):
                temp_file.write(' '.join(
                    random.sample(dialect_symbols, sentence_length)
                ) + '\n')
    return temp_file_name


def generate(args):
    assert_test_corpus_and_vocab_files_specified(args)
    assert args.path is not None, "--path required for generation!"

    print(args)

    # Benchmarking should be language-agnostic
    args.source_lang = "src"
    args.target_lang = "tgt"

    src_dict = pytorch_translate_dictionary.Dictionary.load(args.source_vocab_file)
    dst_dict = pytorch_translate_dictionary.Dictionary.load(args.target_vocab_file)

    # Generate synthetic raw text files
    source_text_file = generate_synthetic_text(
        args.source_lang, src_dict.symbols, args
    )
    target_text_file = generate_synthetic_text(
        args.target_lang, src_dict.symbols, args
    )

    dataset = data.LanguageDatasets(
        src=args.source_lang, dst=args.target_lang, src_dict=src_dict, dst_dict=dst_dict
    )
    models, model_args = pytorch_translate_utils.load_diverse_ensemble_for_inference(
        args.path, dataset.src_dict, dataset.dst_dict
    )
    append_eos_to_source = model_args[0].append_eos_to_source
    reverse_source = model_args[0].reverse_source
    assert all(
        a.append_eos_to_source == append_eos_to_source
        and a.reverse_source == reverse_source
        for a in model_args
    )
    dataset.splits[args.gen_subset] = pytorch_translate_data.make_language_pair_dataset_from_text(
        source_text_file=source_text_file,
        target_text_file=target_text_file,
        source_dict=src_dict,
        target_dict=dst_dict,
        append_eos=append_eos_to_source,
        reverse_source=reverse_source,
    )

    # Remove temporary text files
    os.remove(source_text_file)
    os.remove(target_text_file)

    if args.source_lang is None or args.target_lang is None:
        # record inferred languages in args
        args.source_lang, args.target_lang = dataset.src, dataset.dst

    print(f"| [{dataset.src}] dictionary: {len(dataset.src_dict)} types")
    print(f"| [{dataset.dst}] dictionary: {len(dataset.dst_dict)} types")
    print(f"| {args.gen_subset} {len(dataset.splits[args.gen_subset])} examples")
    args.keep_detailed_timing = True
    scorer, num_sentences, gen_timer, _ = pytorch_translate_generate._generate_score(
        models=models, args=args, dataset=dataset, dataset_split=args.gen_subset
    )

    # Remove contribution of primer sentence
    gen_timer.reset_bucket(0)

    print(
        f"| Translated {num_sentences} sentences ({sum(gen_timer.n)} tokens) "
        f"in {sum(gen_timer.sum):.3f}s ({1. / gen_timer.avg:.2f} tokens/s)"
    )

    for bucket_id in range(gen_timer.n_buckets):
        if gen_timer.n[bucket_id] != 0:
            print(
                "  | Length {}: {} sentences ({} tok) in {:.3f}s ({:.3f} tok/s, avg. latency {:4f}s)".format(
                    bucket_id * args.increment,
                    gen_timer.count[bucket_id],
                    gen_timer.n[bucket_id],
                    gen_timer.sum[bucket_id],
                    1. / gen_timer.avgs[bucket_id],
                    gen_timer.sum[bucket_id] / gen_timer.count[bucket_id],
                )
            )

    print(
        f"| Generate {args.gen_subset} with beam={args.beam}: "
        f"{scorer.result_string()}"
    )
    return scorer.score()


if __name__ == "__main__":
    main()
