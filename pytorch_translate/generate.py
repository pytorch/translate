#!/usr/bin/env python3

import argparse
import collections
import os
import pickle
from typing import List, NamedTuple, Optional

import numpy as np
import torch
from fairseq import (
    data,
    iterative_refinement_generator,
    options,
    progress_bar,
    tasks,
    utils,
)
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.models import FairseqEncoderDecoderModel, FairseqMultiModel
from fairseq.scoring import bleu
from pytorch_translate import (  # noqa  # noqa  # noqa
    beam_decode,
    char_aware_hybrid,
    char_source_hybrid,
    char_source_model,
    char_source_transformer_model,
    hybrid_transformer_rnn,
    options as pytorch_translate_options,
    rnn,
    transformer,
    utils as pytorch_translate_utils,
)
from pytorch_translate.constants import CHECKPOINT_PATHS_DELIMITER
from pytorch_translate.data import data as pytorch_translate_data
from pytorch_translate.dual_learning.dual_learning_models import DualLearningModel
from pytorch_translate.research.beam_search import competing_completed
from pytorch_translate.research.multisource import multisource_data, multisource_decode
from pytorch_translate.tasks.pytorch_translate_multi_task import (
    PyTorchTranslateMultiTask,
)
from pytorch_translate.tasks.pytorch_translate_task import PytorchTranslateTask


try:
    from fairseq.models.fb_levenshtein_transformer import (
        LevenshteinTransformerModel,
    )  # noqa
except ImportError:
    from fairseq.models.nat import LevenshteinTransformerModel


def generate_score(
    args: argparse.Namespace,
    task: tasks.FairseqTask,
    dataset: data.FairseqDataset,
    models: List[FairseqEncoderDecoderModel],
    lang_pair: Optional[str] = None,
    modify_target_dict: bool = True,
):
    """
    Generation for single and multi model training

    Args:
        args: Command-line arguments.
        task: FairseqTask object.
        dataset: Dataset set object for a specific split for a specific model
        models: List[FairseqEncoderDecoderModel], an ensemble of models
        lang_pair: Optional model key in a multi model object. Specify None in
            single model set up
    """
    if lang_pair and len(models) > 0 and isinstance(models[0], FairseqMultiModel):
        if isinstance(dataset, data.RoundRobinZipDatasets):
            dataset = dataset.datasets[lang_pair]
        return _generate_score(
            models=[multi_model.models[lang_pair] for multi_model in models],
            args=args,
            task=task,
            dataset=dataset,
            modify_target_dict=modify_target_dict,
        )
    elif lang_pair and len(models) > 0 and isinstance(models[0], DualLearningModel):
        # TODO: this could be refactored to use lang_pari as key too
        return _generate_score(
            models=[
                multi_model.models["primal"]
                if lang_pair == "primal_parallel"
                else multi_model.models["dual"]
                for multi_model in models
            ],
            args=args,
            task=task,
            dataset=dataset,
            modify_target_dict=modify_target_dict,
        )

    else:
        return _generate_score(
            models=models,
            args=args,
            task=task,
            dataset=dataset,
            modify_target_dict=modify_target_dict,
        )


class TranslationInfo(NamedTuple):
    sample_id: torch.Tensor
    src_tokens: torch.Tensor
    target_tokens: torch.Tensor
    hypo_tokens: torch.Tensor
    src_str: str
    target_str: str
    hypo_str: str
    hypo_score: float
    best_hypo_tokens: Optional[torch.Tensor]
    hypos: List[dict]


def build_sequence_generator(args, task, models):
    use_cuda = torch.cuda.is_available() and not args.cpu
    # Initialize generator
    model_weights = None
    if args.model_weights:
        model_weights = [float(w.strip()) for w in args.model_weights.split(",")]
    use_char_source = (
        isinstance(models[0], char_source_model.CharSourceModel)
        or isinstance(
            models[0], char_source_transformer_model.CharSourceTransformerModel
        )
        or isinstance(models[0], char_source_hybrid.CharSourceHybridModel)
    )
    # Use a different sequence generator in the multisource setting
    if getattr(args, "source_ensembling", False):
        translator_class = multisource_decode.MultiSourceSequenceGenerator
    elif getattr(args, "competing_completed_beam_search", False):
        translator_class = competing_completed.CompetingCompletedSequenceGenerator
    elif isinstance(models[0], LevenshteinTransformerModel):
        translator_class = iterative_refinement_generator.IterativeRefinementGenerator
        return translator_class(tgt_dict=task.target_dictionary, models=models)
    else:
        translator_class = beam_decode.SequenceGenerator
    translator = translator_class(
        models,
        tgt_dict=task.target_dictionary,
        beam_size=args.beam,
        stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized),
        len_penalty=args.length_penalty,
        unk_reward=args.unk_reward,
        word_reward=args.word_reward,
        model_weights=model_weights,
        use_char_source=use_char_source,
        diverse_beam_groups=args.diverse_beam_groups,
        diverse_beam_strength=args.diverse_beam_strength,
        diversity_sibling_gamma=args.diversity_sibling_gamma,
        sampling=args.sampling,
        sampling_topk=args.sampling_topk,
        temperature=args.temperature,
    )
    if use_cuda:
        translator.cuda()
    return translator


def _generate_score(models, args, task, dataset, modify_target_dict):
    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load ensemble
    if not args.quiet:
        print(
            "| loading model(s) from {}".format(
                ", ".join(args.path.split(CHECKPOINT_PATHS_DELIMITER))
            )
        )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=True,
        )

    translator = build_sequence_generator(args, task, models)
    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    print("seed number is" + str(args.max_examples_to_evaluate_seed))
    if args.max_examples_to_evaluate > 0:
        pytorch_translate_data.subsample_pair_dataset(
            dataset, args.max_examples_to_evaluate, args.max_examples_to_evaluate_seed
        )

    # Keep track of translations
    # Initialize with empty translations
    # and zero probs scores
    translated_sentences = [""] * len(dataset)
    translated_scores = [0.0] * len(dataset)
    hypos_list = []

    collect_output_hypos = getattr(args, "output_hypos_binary_path", False)
    if collect_output_hypos:
        output_hypos_token_arrays = [None] * len(dataset)

    # Generate and compute BLEU score
    dst_dict = task.target_dictionary
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer(bleu.SacrebleuConfig())
    else:
        scorer = bleu.Scorer(
            bleu.BleuConfig(
                pad=dst_dict.pad(),
                eos=dst_dict.eos(),
                unk=dst_dict.unk(),
            )
        )

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    oracle_scorer = None
    if args.report_oracle_bleu:
        oracle_scorer = bleu.Scorer(
            bleu.BleuConfig(
                pad=dst_dict.pad(),
                eos=dst_dict.eos(),
                unk=dst_dict.unk(),
            )
        )

    rescorer = None
    num_sentences = 0
    translation_samples = []
    translation_info_list = []
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        gen_timer = StopwatchMeter()
        translations = translator.generate_batched_itr(
            t,
            maxlen_a=args.max_len_a,
            maxlen_b=args.max_len_b,
            cuda=use_cuda,
            timer=gen_timer,
            prefix_size=1
            if pytorch_translate_data.is_multilingual_many_to_one(args)
            else 0,
        )

        for trans_info in _iter_translations(
            args, task, dataset, translations, align_dict, rescorer, modify_target_dict
        ):
            if hasattr(scorer, "add_string"):
                scorer.add_string(trans_info.target_str, trans_info.hypo_str)
            else:
                scorer.add(trans_info.target_tokens, trans_info.hypo_tokens)
            if oracle_scorer is not None:
                oracle_scorer.add(trans_info.target_tokens, trans_info.best_hypo_tokens)

            if getattr(args, "translation_output_file", False):
                translated_sentences[trans_info.sample_id] = trans_info.hypo_str
            if getattr(args, "translation_probs_file", False):
                translated_scores[trans_info.sample_id] = trans_info.hypo_score
            if getattr(args, "hypotheses_export_path", False):
                hypos_list.append(trans_info.hypos)
            if collect_output_hypos:
                output_hypos_token_arrays[
                    trans_info.sample_id
                ] = trans_info.best_hypo_tokens
            if args.translation_info_export_path is not None:
                # Strip expensive data from hypotheses before saving
                hypos = [
                    {k: v for k, v in hypo.items() if k in ["tokens", "score"]}
                    for hypo in trans_info.hypos
                ]
                # Make sure everything is on cpu before exporting
                hypos = [
                    {"score": hypo["score"], "tokens": hypo["tokens"].cpu()}
                    for hypo in hypos
                ]
                translation_info_list.append(
                    {
                        "src_tokens": trans_info.src_tokens.cpu(),
                        "target_tokens": trans_info.target_tokens,
                        "hypos": hypos,
                    }
                )
            translation_samples.append(
                collections.OrderedDict(
                    {
                        "sample_id": trans_info.sample_id.item(),
                        "src_str": trans_info.src_str,
                        "target_str": trans_info.target_str,
                        "hypo_str": trans_info.hypo_str,
                    }
                )
            )
            wps_meter.update(trans_info.src_tokens.size(0))
            t.log({"wps": round(wps_meter.avg)})
            num_sentences += 1

    # If applicable, save collected hypothesis tokens to binary output file
    if collect_output_hypos:
        output_dataset = pytorch_translate_data.InMemoryIndexedDataset()
        output_dataset.load_from_sequences(output_hypos_token_arrays)
        output_dataset.save(args.output_hypos_binary_path)
    if args.output_source_binary_path:
        dataset.src.save(args.output_source_binary_path)
    if args.translation_info_export_path is not None:
        f = open(args.translation_info_export_path, "wb")
        pickle.dump(translation_info_list, f)
        f.close()

    # If applicable, save the translations and scores to the output files
    # These two ouputs are used in dual learning for weighted backtranslation
    if getattr(args, "translation_output_file", False) and getattr(
        args, "translation_probs_file", False
    ):
        with open(args.translation_output_file, "w") as translation_file, open(
            args.translation_probs_file, "w"
        ) as score_file:
            for hypo_str, hypo_score in zip(translated_sentences, translated_scores):
                if len(hypo_str.strip()) > 0:
                    print(hypo_str, file=translation_file)
                    print(np.exp(hypo_score), file=score_file)

    # For eg. external evaluation
    if getattr(args, "hypotheses_export_path", False):
        with open(args.hypotheses_export_path, "w") as out_file:
            for hypos in hypos_list:
                for hypo in hypos:
                    print(
                        task.tgt_dict.string(
                            hypo["tokens"], bpe_symbol=args.post_process
                        ),
                        file=out_file,
                    )

    if oracle_scorer is not None:
        print(f"| Oracle BLEU (best hypo in beam): {oracle_scorer.result_string()}")

    return scorer, num_sentences, gen_timer, translation_samples


def smoothed_sentence_bleu(task, target_tokens, hypo_tokens):
    """
    Implements "Smoothing 3" method from Chen and Cherry. "A Systematic
    Comparison of Smoothing Techniques for Sentence-Level BLEU".
    http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    """
    dst_dict = task.target_dictionary
    scorer = bleu.Scorer(
        bleu.BleuConfig(
            pad=dst_dict.pad(),
            eos=dst_dict.eos(),
            unk=dst_dict.unk(),
        )
    )
    scorer.add(target_tokens, hypo_tokens)

    invcnt = 1
    ratios = []
    for (match, count) in [
        (scorer.stat.match1, scorer.stat.count1),
        (scorer.stat.match2, scorer.stat.count2),
        (scorer.stat.match3, scorer.stat.count3),
        (scorer.stat.match4, scorer.stat.count4),
    ]:
        if count == 0:
            # disregard n-grams for values of n larger than hypothesis length
            continue
        if match == 0:
            invcnt *= 2
            match = 1.0 / invcnt
        ratios.append(match / count)

    brevity_penalty = np.min(
        [1, np.exp(1 - (scorer.stat.reflen / scorer.stat.predlen))]
    )
    geometric_mean = np.exp(np.log(ratios).mean())
    smoothed_bleu = brevity_penalty * geometric_mean
    return smoothed_bleu


def _iter_translations(
    args, task, dataset, translations, align_dict, rescorer, modify_target_dict
):
    """Iterate over translations.

    This is a generator function which wraps the beam-search sequence generator,
    performing such work on the output as converting token indices to
    strings, printing output where applicable (not args.quiet), collecting
    oracle translations where applicable, and removing language-ID tokens
    for multilingual translation.

    Args:
        args: Command-line arguments.
        task: FairseqTask object.
        dataset: Dataset set object for a specific split.
        translations: Batched translation iterator, as returned by
            SequenceGenerator.generate_batched_itr().
        align_dict: Dictionary for UNK replacement.

    Yields:
        For each sentence in `translations`, yields a TranslationInfo.
    """
    is_multilingual = pytorch_translate_data.is_multilingual_many_to_one(args)

    for sample_id, src_tokens, target_tokens, hypos in translations:
        # Process input and ground truth
        target_tokens = target_tokens.int().cpu()

        if is_multilingual:
            src_lang_id = (
                src_tokens[-1] - pytorch_translate_data.MULTILING_DIALECT_ID_OFFSET
            )
            target_lang_id = (
                target_tokens[0] - pytorch_translate_data.MULTILING_DIALECT_ID_OFFSET
            )

            # remove language ID tokens
            src_tokens = src_tokens[:-1]
            target_tokens = target_tokens[1:]

            # Select dictionaries
            src_dict = task.source_dictionaries[task.get_encoder_lang_code(src_lang_id)]
            target_dict = task.target_dictionaries[
                task.get_decoder_lang_code(target_lang_id)
            ]
        else:
            src_dict = task.source_dictionary
            target_dict = task.target_dictionary

        # Either retrieve the original sentences or regenerate them from tokens.
        if align_dict is not None:
            src_str = dataset.src.get_original_text(sample_id)
            target_str = dataset.tgt.get_original_text(sample_id)
        else:
            src_str = src_dict.string(src_tokens, args.post_process)
            target_str = target_dict.string(
                target_tokens, args.post_process, escape_unk=True
            )

        if not args.quiet:
            print(f"S-{sample_id}\t{src_str}")
            print(f"T-{sample_id}\t{target_str}")

        # used for oracle evaluation (args.report_oracle_bleu)
        best_hypo_tokens = None
        best_hypo_score = 0
        collect_oracle_hypos = args.report_oracle_bleu or (
            args.output_hypos_binary_path and args.nbest > 0
        )

        # Process top predictions
        for i, hypo in enumerate(hypos[: min(len(hypos), args.nbest)]):
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str=src_str,
                alignment=hypo["alignment"].int().cpu()
                if align_dict is not None
                else None,
                align_dict=align_dict,
                tgt_dict=task.target_dictionary,
                remove_bpe=args.post_process,
            )

            if not args.quiet:
                print(f"H-{sample_id}\t{hypo['score']}\t{hypo_str}")
                if alignment is not None:
                    print(
                        "A-{}\t{}".format(
                            sample_id,
                            " ".join(map(lambda x: str(utils.item(x)), alignment)),
                        )
                    )

            if collect_oracle_hypos:
                score = smoothed_sentence_bleu(task, target_tokens, hypo_tokens)
                if score > best_hypo_score:
                    best_hypo_tokens = hypo_tokens
                    best_hypo_score = score

            if i == 0:
                if align_dict is not None or args.post_process is not None:
                    # Convert back to tokens for evaluation with unk replacement
                    # and/or without BPE
                    target_tokens = task.target_dictionary.encode_line(
                        target_str, add_if_not_exist=modify_target_dict
                    )
                # The probs score for the hypo_str; whether it's normalized by
                # sequence length or not depends on normalize_scores, which is
                # set by arg.nonormalize.
                # However, as I tried, whether normalize_scores is set or not,
                # the returned scores are the same (to be investigated).
                # Here, the probs are normalized by hypo length so the value
                # is big enough to be used as weights for backtranslations in
                # dual learning.
                hypo_score = (
                    hypo["score"] / len(hypo_tokens) if len(hypo_tokens) > 0 else 0.0
                )
                top_hypo_tokens = hypo_tokens
                top_hypo_str = hypo_str

        if not collect_oracle_hypos:
            best_hypo_tokens = top_hypo_tokens

        yield TranslationInfo(
            sample_id=sample_id,
            src_tokens=src_tokens,
            target_tokens=target_tokens,
            hypo_tokens=top_hypo_tokens,
            src_str=src_str,
            target_str=target_str,
            hypo_str=top_hypo_str,
            hypo_score=hypo_score,
            best_hypo_tokens=best_hypo_tokens,
            hypos=hypos,
        )


def add_args(parser):
    group = parser.add_argument_group("Generation")
    group.add_argument(
        "--word-reward",
        type=float,
        default=0.0,
        help=(
            "Value to add to (log-prob) score for each token except EOS. "
            "IMPORTANT NOTE: higher values of --length-penalty and --word-reward "
            "both encourage longer translations, while higher values of "
            "--unk-reward penalize UNKs more."
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
    parser = options.get_parser("Generation", default_task="pytorch_translate")
    pytorch_translate_options.add_verbosity_args(parser)
    pytorch_translate_options.add_dataset_args(parser, gen=True)
    generation_group = options.add_generation_args(parser)
    pytorch_translate_options.expand_generation_args(generation_group)

    # Adds args used by the standalone generate binary.
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
            "Same as --target-vocab-file except using characters. "
            "(For use with char_aware models only.)"
        ),
    )
    generation_group.add_argument(
        "--source-text-file",
        default="",
        nargs="+",
        metavar="FILE",
        help="Path to raw text file containing examples in source dialect. "
        "This overrides what would be loaded from the data dir. "
        "You can specify multiple source files (eg. for use in combination "
        "with --source-ensembling). By default this will only translate the "
        "first source file",
    )
    generation_group.add_argument(
        "--target-text-file",
        default="",
        metavar="FILE",
        help="Path to raw text file containing examples in target dialect. "
        "This overrides what would be loaded from the data dir.",
    )
    generation_group.add_argument(
        "--source-binary-file",
        default="",
        help="Path for the binary file containing source eval examples. "
        "(Overrides --source-text-file. Must be used in conjunction with "
        "--target-binary-file).",
    )
    generation_group.add_argument(
        "--target-binary-file",
        default="",
        help="Path for the binary file containing target eval examples. "
        "(Overrides --target-text-file. Must be used in conjunction with "
        "--source-binary-file).",
    )
    generation_group.add_argument(
        "--translation-output-file",
        default="",
        type=str,
        metavar="FILE",
        help="Path to text file to store the output of the model. ",
    )
    generation_group.add_argument(
        "--translation-probs-file",
        default="",
        type=str,
        metavar="FILE",
        help="Path to text file to store the probs of translation output. ",
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
    generation_group.add_argument(
        "--source-ensembling",
        action="store_true",
        help="If this flag is present, the model will ensemble the predictions "
        "conditioned on multiple source sentences (one per source-text-file)",
    )
    generation_group.add_argument(
        "--competing-completed-beam-search",
        action="store_true",
        help="If this flag is present, use the alternative beam search "
        "implementation in research/beam_search. This beam search keeps completed "
        "hypos in the beam and let them compete against hypo expansions in the "
        "next time step.",
    )
    generation_group.add_argument(
        "--iterative-refinement",
        action="store_true",
        help="If this flag is present, use iterative refinement.",
    )

    return parser


def main():
    parser = get_parser_with_args()
    args = options.parse_args_and_arch(parser)
    validate_args(args)
    generate(args)


def validate_args(args):
    pytorch_translate_options.validate_generation_args(args)

    assert args.path is not None, "--path required for generation!"
    if args.source_binary_file != "":
        assert args.target_binary_file != ""
        assert os.path.isfile(args.source_binary_file)
        assert os.path.isfile(args.target_binary_file)
    else:
        assert all(
            (src_file and os.path.isfile(src_file))
            for src_file in args.source_text_file
        ), "Please specify a valid file for --source-text-file"
        assert args.target_text_file and os.path.isfile(
            args.target_text_file
        ), "Please specify a valid file for --target-text-file"


def generate(args):
    pytorch_translate_options.print_args(args)

    (
        models,
        model_args,
        task,
    ) = pytorch_translate_utils.load_diverse_ensemble_for_inference(
        args.path.split(CHECKPOINT_PATHS_DELIMITER)
    )
    args.source_lang = model_args[0].source_lang
    args.target_lang = model_args[0].target_lang

    append_eos_to_source = model_args[0].append_eos_to_source
    reverse_source = model_args[0].reverse_source
    assert all(
        a.append_eos_to_source == append_eos_to_source
        and a.reverse_source == reverse_source
        for a in model_args
    )
    if args.source_binary_file != "":
        assert args.target_binary_file != ""
        if isinstance(task, PytorchTranslateTask):
            task.load_dataset(
                args.gen_subset,
                args.source_binary_file,
                args.target_binary_file,
                is_npz=args.is_npz,
            )
        else:
            task.load_dataset(
                args.gen_subset, args.source_binary_file, args.target_binary_file
            )
    elif pytorch_translate_data.is_multilingual_many_to_one(args):
        task.set_encoder_langs(model_args[0].multiling_encoder_lang)
        task.set_decoder_langs(model_args[0].multiling_decoder_lang)
        task.load_dataset_from_text_multilingual(
            args.gen_subset,
            source_text_file=args.source_text_file[0],
            target_text_file=args.target_text_file,
            source_lang_id=task.get_encoder_lang_id(args.multiling_source_lang[0]),
            target_lang_id=task.get_decoder_lang_id(args.multiling_target_lang[0]),
            append_eos=append_eos_to_source,
            reverse_source=reverse_source,
        )
    elif args.source_ensembling:
        task.load_multisource_dataset_from_text(
            args.gen_subset,
            source_text_files=args.source_text_file,
            target_text_file=args.target_text_file,
            append_eos=append_eos_to_source,
            reverse_source=reverse_source,
        )
    else:
        task.load_dataset_from_text(
            args.gen_subset,
            source_text_file=args.source_text_file[0],
            target_text_file=args.target_text_file,
            append_eos=append_eos_to_source,
            reverse_source=reverse_source,
        )

    lang_pair = None
    if isinstance(task, PyTorchTranslateMultiTask):
        if args.source_lang and args.target_lang:
            lang_pair = args.source_lang + "-" + args.target_lang
        else:
            lang_pair = "src-tgt"
    scorer, num_sentences, gen_timer, _ = generate_score(
        args=args,
        task=task,
        dataset=task.dataset(args.gen_subset),
        lang_pair=lang_pair,
        models=models,
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
