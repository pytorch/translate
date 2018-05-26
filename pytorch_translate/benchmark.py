#!/usr/bin/env python3

import os
import random
import time
import torch

from fairseq import bleu, data, options, progress_bar, tokenizer, utils
from fairseq.meters import TimeMeter
from pytorch_translate import beam_decode
from pytorch_translate import data as pytorch_translate_data
from pytorch_translate import dictionary as pytorch_translate_dictionary
from pytorch_translate import rnn  # noqa


# Variation on the fairseq StopwatchMeter that separates statistics by number
# of tokens. Sentences longer than max_length are stored in the last bucket.
class BucketStopwatchMeter(object):
    def __init__(self, increment, max_length, sentences_per_batch):
        self.increment = increment
        self.n_buckets = max_length // increment + 1
        self.sentences_per_batch = sentences_per_batch
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            bucket_id = min(
                self.n_buckets - 1,
                n // self.increment,
            )
            self.sum[bucket_id] += delta
            self.n[bucket_id] += n
            self.count[bucket_id] += 1
            self.start_time = None

    def reset(self):
        self.sum = [0] * self.n_buckets
        self.n = [0] * self.n_buckets
        self.count = [0] * self.n_buckets
        self.start_time = None

    def reset_bucket(self, bucket_id):
        if self.start_time is None:
            self.sum[bucket_id] = 0
            self.n[bucket_id] = 0
            self.count[bucket_id] = 0

    @property
    def avg(self):
        return sum(self.sum) / sum(self.n)

    @property
    def avgs(self):
        result = [0] * self.n_buckets
        for i in range(self.n_buckets):
            if self.n[i] != 0:
                result[i] = self.sum[i] / self.n[i]
            else:
                result[i] = 0
        return result


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
    model_weights = None
    if args.model_weights:
        model_weights = [float(w.strip()) for w in args.model_weights.split(",")]
    translator = beam_decode.SequenceGenerator(
        models,
        beam_size=args.beam,
        stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized),
        len_penalty=args.lenpen,
        unk_penalty=args.unkpen,
        word_reward=args.word_reward,
        model_weights=model_weights,
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
        gen_timer = BucketStopwatchMeter(
            args.increment,
            args.max_length,
            args.samples_per_length,
        )
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


def load_diverse_ensemble_for_inference(filenames, src_dict, dst_dict):
    """Load an ensemble of diverse models for inference.

    This method is similar to fairseq.utils.load_ensemble_for_inference
    but allows to load diverse models with non-uniform args.

    Args:
        filenames: List of file names to checkpoints
        src_dict: Source dictionary
        dst_dict: Target dictionary

    Return:
        models, args: Tuple of lists. models contains the loaded models, args
        the corresponding configurations.
    """
    from fairseq import models

    # load model architectures and weights
    states = []
    for filename in filenames:
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))
        states.append(
            torch.load(
                filename,
                map_location=lambda s, l: torch.serialization.default_restore_location(
                    s, "cpu"
                ),
            )
        )
    # build ensemble
    ensemble = []
    for state in states:
        model = models.build_model(state["args"], src_dict, dst_dict)
        model.load_state_dict(state["model"])
        ensemble.append(model)
    return ensemble, [s["args"] for s in states]


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
    models, model_args = load_diverse_ensemble_for_inference(
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
    scorer, num_sentences, gen_timer = _generate_score(
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
