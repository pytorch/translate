#!/usr/bin/env python3

import os
import random
import tempfile

from fairseq import options, tasks
from pytorch_translate import (  # noqa; noqa
    generate as pytorch_translate_generate,
    options as pytorch_translate_options,
    rnn,
    utils as pytorch_translate_utils,
)
from pytorch_translate.constants import CHECKPOINT_PATHS_DELIMITER


def get_parser_with_args():
    parser = options.get_parser("Generation", default_task="pytorch_translate")
    pytorch_translate_options.add_verbosity_args(parser)
    pytorch_translate_options.add_dataset_args(parser, gen=True)
    generation_group = options.add_generation_args(parser)
    pytorch_translate_options.expand_generation_args(generation_group)

    generation_group.add_argument(
        "--source-vocab-file",
        default="",
        metavar="FILE",
        help="Path to text file representing the Dictionary to use.",
    )
    generation_group.add_argument(
        "--char-source-vocab-file",
        default="",
        metavar="FILE",
        help=(
            "Same as --source-vocab-file except using characters. "
            "(For use with char_source and char_aware models only.)"
        ),
    )
    generation_group.add_argument(
        "--target-vocab-file",
        default="",
        metavar="FILE",
        help="Path to text file representing the Dictionary to use.",
    )
    generation_group.add_argument(
        "--char-target-vocab-file",
        default="",
        metavar="FILE",
        help=(
            "Same as --source-target-file except using characters. "
            "(For use with char_aware models only.)"
        ),
    )
    generation_group.add_argument(
        "--multiling-source-lang",
        action="append",
        metavar="SRC",
        help=(
            "Must be set for decoding with multilingual models. "
            "Must match an entry from --multiling-encoder-lang from training."
        ),
    )
    generation_group.add_argument(
        "--multiling-target-lang",
        action="append",
        metavar="TARGET",
        help=(
            "Must be set for decoding with multilingual models. "
            "Must match an entry from --multiling-decoder-lang from training."
        ),
    )

    # Add args related to benchmarking.
    group = parser.add_argument_group("Benchmarking")
    group.add_argument(
        "--runs-per-length",
        default=10,
        type=int,
        help="Number of times to run generation on each length.",
    )
    group.add_argument(
        "--examples-per-length",
        default=1,
        type=int,
        help="Sentences of each length to include in each eval (batched if >1).",
    )

    return parser


def main():
    parser = get_parser_with_args()
    # args = parser.parse_args()
    args = options.parse_args_and_arch(parser)
    # Disable printout of all source and target sentences
    args.quiet = True
    benchmark(args)


def generate_synthetic_text(dialect, dialect_symbols, length, examples):
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, dir="/tmp")
    temp_file_name = temp_file.name
    temp_file.close()
    with open(temp_file_name, "w") as temp_file:
        for _ in range(examples):
            temp_file.write(" ".join(random.sample(dialect_symbols, length)) + "\n")
    return temp_file_name


def benchmark(args):
    assert args.source_vocab_file and os.path.isfile(
        args.source_vocab_file
    ), "Please specify a valid file for --source-vocab-file"
    assert args.target_vocab_file and os.path.isfile(
        args.target_vocab_file
    ), "Please specify a valid file for --target-vocab_file"
    assert args.path is not None, "--path required for generation!"

    print(args)

    # Benchmarking should be language-agnostic
    args.source_lang = "src"
    args.target_lang = "tgt"

    (
        models,
        model_args,
        task,
    ) = pytorch_translate_utils.load_diverse_ensemble_for_inference(
        args.path.split(CHECKPOINT_PATHS_DELIMITER)
    )

    append_eos_to_source = model_args[0].append_eos_to_source
    reverse_source = model_args[0].reverse_source
    assert all(
        a.append_eos_to_source == append_eos_to_source
        and a.reverse_source == reverse_source
        for a in model_args
    )

    def benchmark_length(n):
        # Generate synthetic raw text files
        source_text_file = generate_synthetic_text(
            dialect=args.source_lang,
            dialect_symbols=task.source_dictionary.symbols,
            length=n,
            examples=args.examples_per_length,
        )
        target_text_file = generate_synthetic_text(
            dialect=args.target_lang,
            dialect_symbols=task.target_dictionary.symbols,
            length=n,
            examples=args.examples_per_length,
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

        # priming
        scorer, num_sentences, gen_timer, _ = pytorch_translate_generate.generate_score(
            models=models, args=args, task=task, dataset=task.dataset(args.gen_subset)
        )

        total_time = 0.0
        for _ in range(args.runs_per_length):
            (
                scorer,
                num_sentences,
                gen_timer,
                _,
            ) = pytorch_translate_generate.generate_score(
                models=models,
                args=args,
                task=task,
                dataset=task.dataset(args.gen_subset),
            )
            total_time += gen_timer.sum
            gen_timer.reset()

        sentences_per_run = args.examples_per_length
        runs = args.runs_per_length
        total_sentences = sentences_per_run * runs
        total_tokens = total_sentences * n

        print(f"--- {n} tokens ---")
        print(f"Generated {total_tokens} tokens ({runs} runs of {sentences_per_run})")
        print(f"Total time: {total_time:.3f} seconds")
        time_per_sentence = total_time / total_sentences
        print(f"Time per sentence: {time_per_sentence:.3f} seconds\n")

    benchmark_length(6)
    benchmark_length(10)
    benchmark_length(20)


if __name__ == "__main__":
    main()
