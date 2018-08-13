#!/usr/bin/env python3

import os
import random

import torch
from fairseq import options, tasks
from fairseq.meters import TimeMeter
from pytorch_translate import (
    beam_decode,
    data as pytorch_translate_data,
    dictionary as pytorch_translate_dictionary,
    generate as pytorch_translate_generate,
    tasks as pytorch_translate_tasks,
    utils as pytorch_translate_utils,
)


from pytorch_translate import rnn  # noqa; noqa


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
        "Must be integer >=1.",
    )
    group.add_argument(
        "--max-length",
        default=100,
        type=int,
        help="Maximum allowed length for synthesized sentences. "
        "Should be greater than --increment.",
    )
    group.add_argument(
        "--samples-per-length",
        default=1,
        type=int,
        help="Number of sentences to be synthesized at each length. ",
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
    assert args.source_vocab_file and os.path.isfile(
        args.source_vocab_file
    ), "Please specify a valid file for --source-vocab-file"
    assert args.target_vocab_file and os.path.isfile(
        args.target_vocab_file
    ), "Please specify a valid file for --target-vocab_file"


def generate_synthetic_text(dialect, dialect_symbols, args):
    assert args.max_length >= 1, "Please specify a valid maximum length"
    temp_file_name = f"benchmark_text_file_{dialect}"
    with open(temp_file_name, "w") as temp_file:
        # Short sentence to prime GPU
        temp_file.write(dialect_symbols[0] + "\n")

        for _ in range(args.samples_per_length):
            for sentence_length in range(
                args.increment, args.max_length, args.increment
            ):
                temp_file.write(
                    " ".join(random.sample(dialect_symbols, sentence_length)) + "\n"
                )
    return temp_file_name


def generate(args):
    assert_test_corpus_and_vocab_files_specified(args)
    assert args.path is not None, "--path required for generation!"

    print(args)

    # Benchmarking should be language-agnostic
    args.source_lang = "src"
    args.target_lang = "tgt"

    task = tasks.setup_task(args)
    models, model_args = pytorch_translate_utils.load_diverse_ensemble_for_inference(
        args.path.split(":"), task
    )

    # Generate synthetic raw text files
    source_text_file = generate_synthetic_text(
        args.source_lang, task.source_dictionary.symbols, args
    )
    target_text_file = generate_synthetic_text(
        args.target_lang, task.target_dictionary.symbols, args
    )

    append_eos_to_source = model_args[0].append_eos_to_source
    reverse_source = model_args[0].reverse_source
    assert all(
        a.append_eos_to_source == append_eos_to_source
        and a.reverse_source == reverse_source
        for a in model_args
    )

    task.load_dataset_from_text(
        args.gen_subset,
        source_text_file=source_text_file,
        target_text_file=target_text_file,
        append_eos=append_eos_to_source,
        reverse_source=reverse_source,
    )

    # Remove temporary text files
    os.remove(source_text_file)
    os.remove(target_text_file)

    args.keep_detailed_timing = True
    scorer, num_sentences, gen_timer, _ = pytorch_translate_generate._generate_score(
        models=models, args=args, task=task, dataset_split=args.gen_subset
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
