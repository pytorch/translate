#!/usr/bin/env python3

import os
import torch

from fairseq import bleu, data, options, progress_bar, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from pytorch_translate import beam_decode
from pytorch_translate import data as pytorch_translate_data
from pytorch_translate import dictionary as pytorch_translate_dictionary
from pytorch_translate import rnn  # noqa


def generate_score(args, dataset, dataset_split):
    models, _ = utils.load_ensemble_for_inference(
        args.path, dataset.src_dict, dataset.dst_dict
    )
    return _generate_score(models, args, dataset, dataset_split)


def _generate_score(models, args, dataset, dataset_split):
    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load ensemble
    if not args.quiet:
        print("| loading model(s) from {}".format(", ".join(args.path)))

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam
        )

    # Initialize generator
    translator = beam_decode.SequenceGenerator(
        models,
        beam_size=args.beam,
        stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized),
        len_penalty=args.lenpen,
        unk_penalty=args.unkpen,
        word_reward=args.word_reward,
    )
    if use_cuda:
        translator.cuda()
    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Generate and compute BLEU score
    scorer = bleu.Scorer(
        dataset.dst_dict.pad(), dataset.dst_dict.eos(), dataset.dst_dict.unk()
    )
    max_positions = min(model.max_encoder_positions() for model in models)
    itr = dataset.eval_dataloader(
        dataset_split,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        skip_invalid_size_inputs_valid_test=(args.skip_invalid_size_inputs_valid_test),
    )
    if args.num_shards > 1:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise ValueError("--shard-id must be between 0 and num_shards")
        itr = data.sharded_iterator(itr, args.num_shards, args.shard_id)

    num_sentences = 0
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        gen_timer = StopwatchMeter()
        translations = translator.generate_batched_itr(
            t,
            maxlen_a=args.max_len_a,
            maxlen_b=args.max_len_b,
            cuda=use_cuda,
            timer=gen_timer,
        )
        for sample_id, src_tokens, target_tokens, hypos in translations:
            # Process input and ground truth
            target_tokens = target_tokens.int().cpu()
            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = dataset.splits[dataset_split].src.get_original_text(sample_id)
                target_str = dataset.splits[dataset_split].dst.get_original_text(
                    sample_id
                )
            else:
                src_str = dataset.src_dict.string(src_tokens, args.remove_bpe)
                target_str = dataset.dst_dict.string(
                    target_tokens, args.remove_bpe, escape_unk=True
                )

            if not args.quiet:
                print(f"S-{sample_id}\t{src_str}")
                print(f"T-{sample_id}\t{target_str}")

            # Process top predictions
            for i, hypo in enumerate(hypos[: min(len(hypos), args.nbest)]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"].int().cpu(),
                    align_dict=align_dict,
                    dst_dict=dataset.dst_dict,
                    remove_bpe=args.remove_bpe,
                )

                if not args.quiet:
                    print(f"H-{sample_id}\t{hypo['score']}\t{hypo_str}")
                    print(
                        "A-{}\t{}".format(
                            sample_id,
                            " ".join(map(lambda x: str(utils.item(x)), alignment)),
                        )
                    )

                # Score only the top hypothesis
                if i == 0:
                    if align_dict is not None or args.remove_bpe is not None:
                        # Convert back to tokens for evaluation with unk replacement
                        # and/or without BPE
                        target_tokens = tokenizer.Tokenizer.tokenize(
                            target_str, dataset.dst_dict, add_if_not_exist=True
                        )
                    scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(src_tokens.size(0))
            t.log({"wps": round(wps_meter.avg)})
            num_sentences += 1

    return scorer, num_sentences, gen_timer


def add_args(parser):
    group = parser.add_argument_group("Generation")
    group.add_argument(
        "--append-eos-to-source",
        action="store_true",
        default=False,
        help=(
            "Apppend EOS to source sentences (instead of just target). When "
            "running generate.py we ignore this command line argument and use "
            "the same --append-eos-to-source value from training."
        ),
    )
    group.add_argument(
        "--reverse-source",
        action="store_true",
        default=True,
        help=(
            "Feed source sentence to model in reverse order. When running "
            "generate.py we ignore this command line argument and use the same "
            "--reverse-source value from training."
        ),
    )
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


def get_parser_with_args():
    parser = options.get_parser("Generation")
    options.add_dataset_args(parser, gen=True)
    options.add_generation_args(parser)
    add_args(parser)

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

    return parser


def main():
    parser = get_parser_with_args()
    args = parser.parse_args()
    generate(args)


def assert_test_corpus_and_vocab_files_specified(args):
    assert not args.data, (
        "Specifying a data directory is disabled in FBTranslate since the "
        "fairseq data class is not supported. Please specify "
        "--train-source-text-file, --train-target-text-file, "
        "--eval-source-text-file, and  --eval-target-text-file instead."
    )
    assert (
        args.source_vocab_file and os.path.isfile(args.source_vocab_file)
    ), "Please specify a valid file for --source-vocab-file"
    assert (
        args.target_vocab_file and os.path.isfile(args.target_vocab_file)
    ), "Please specify a valid file for --target-vocab_file"
    assert (
        args.source_text_file and os.path.isfile(args.source_text_file)
    ), "Please specify a valid file for --source-text-file"
    assert (
        args.target_text_file and os.path.isfile(args.target_text_file)
    ), "Please specify a valid file for --target-text-file"


def generate(args):
    assert_test_corpus_and_vocab_files_specified(args)
    assert args.path is not None, "--path required for generation!"

    print(args)

    if args.source_lang is None:
        args.source_lang = "src"
    if args.target_lang is None:
        args.target_lang = "tgt"

    src_dict = pytorch_translate_dictionary.Dictionary.load(args.source_vocab_file)
    dst_dict = pytorch_translate_dictionary.Dictionary.load(args.target_vocab_file)
    dataset = data.LanguageDatasets(
        src=args.source_lang, dst=args.target_lang, src_dict=src_dict, dst_dict=dst_dict
    )
    models, model_args = utils.load_ensemble_for_inference(
        args.path, dataset.src_dict, dataset.dst_dict
    )
    dataset.splits[args.gen_subset] = pytorch_translate_data.make_language_pair_dataset(
        source_file=args.source_text_file,
        target_file=args.target_text_file,
        source_dict=src_dict,
        target_dict=dst_dict,
        append_eos=model_args.append_eos_to_source,
        reverse_source=model_args.reverse_source,
    )

    if args.source_lang is None or args.target_lang is None:
        # record inferred languages in args
        args.source_lang, args.target_lang = dataset.src, dataset.dst

    print(f"| [{dataset.src}] dictionary: {dataset.src_dict} types")
    print(f"| [{dataset.dst}] dictionary: {len(dataset.dst_dict)} types")
    print(f"| {args.gen_subset} {len(dataset.splits[args.gen_subset])} examples")
    scorer, num_sentences, gen_timer = _generate_score(
        models=models, args=args, dataset=dataset, dataset_split=args.gen_subset
    )
    print(
        f"| Translated {num_sentences} sentences ({gen_timer.n} tokens) "
        f"in {gen_timer.sum:.1f}s ({1. / gen_timer.avg:.2f} tokens/s)"
    )
    print(
        f"| Generate {args.gen_subset} with beam={args.beam}: "
        f"{scorer.result_string()}"
    )
    return scorer.score()


if __name__ == "__main__":
    main()
