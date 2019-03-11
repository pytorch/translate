#!/usr/bin/env python3

import argparse
from typing import List, NamedTuple, Optional

import numpy as np
import pandas as pd
import sacrebleu


def get_sufficient_stats(
    translations: List[str], references: List[str]
) -> pd.DataFrame:
    assert len(translations) == len(references), (
        f"There are {len(translations)} translated sentences "
        f"but {len(references)} reference sentences"
    )
    assert sacrebleu.NGRAM_ORDER == 4, (
        f"Expected SacreBLEU to be using n-gram order 4 "
        f"instead of {sacrebleu.NGRAM_ORDER}."
    )

    sufficient_stats: List[List[int]] = []
    for sentence, ref in zip(translations, references):
        sentence_bleu = sacrebleu.corpus_bleu(
            sys_stream=sentence,
            ref_streams=ref,
            lowercase=False,
            tokenize="none",
            use_effective_order=False,
        )
        sufficient_stats.append(
            [
                # Number of correct 1-grams, .., 4-grams
                sentence_bleu.counts[0],
                sentence_bleu.counts[1],
                sentence_bleu.counts[2],
                sentence_bleu.counts[3],
                # Total number of 1-grams, .., 4-grams
                sentence_bleu.totals[0],
                sentence_bleu.totals[1],
                sentence_bleu.totals[2],
                sentence_bleu.totals[3],
                # Length of translated sentence.
                sentence_bleu.sys_len,
                # Length of reference sentence.
                sentence_bleu.ref_len,
            ]
        )
    return pd.DataFrame(
        sufficient_stats,
        columns=[
            "correct_1_grams",
            "correct_2_grams",
            "correct_3_grams",
            "correct_4_grams",
            "total_1_grams",
            "total_2_grams",
            "total_3_grams",
            "total_4_grams",
            "translation_length",
            "reference_length",
        ],
    )


def calc_bleu_from_stats(sentence_stats: pd.DataFrame) -> sacrebleu.BLEU:
    corpus_stats = sentence_stats.sum(axis=0)
    corpus_bleu = sacrebleu.compute_bleu(
        correct=[
            corpus_stats.correct_1_grams,
            corpus_stats.correct_2_grams,
            corpus_stats.correct_3_grams,
            corpus_stats.correct_4_grams,
        ],
        total=[
            corpus_stats.total_1_grams,
            corpus_stats.total_2_grams,
            corpus_stats.total_3_grams,
            corpus_stats.total_4_grams,
        ],
        sys_len=corpus_stats.translation_length,
        ref_len=corpus_stats.reference_length,
    )
    return corpus_bleu


class PairedBootstrapOutput(NamedTuple):
    baseline_bleu: sacrebleu.BLEU
    new_bleu: sacrebleu.BLEU
    num_samples: int
    # Number of samples where the baseline was better than the new.
    baseline_better: int
    # Number of samples where the baseline and new had identical BLEU score.
    num_equal: int
    # Number of samples where the new was better than baseline.
    new_better: int


def paired_bootstrap_resample(
    baseline_stats: pd.DataFrame,
    new_stats: pd.DataFrame,
    num_samples: int = 1000,
    sample_size: Optional[int] = None,
) -> PairedBootstrapOutput:
    """
    From http://aclweb.org/anthology/W04-3250
    Statistical significance tests for machine translation evaluation (Koehn, 2004)
    """
    assert len(baseline_stats) == len(new_stats), (
        f"Length mismatch - baseline has {len(baseline_stats)} lines "
        f"while new has {len(new_stats)} lines."
    )
    num_sentences = len(baseline_stats)
    if not sample_size:
        # Defaults to sampling new corpora of the same size as the original.
        # This is not identical to the original corpus since we are sampling
        # with replacement.
        sample_size = num_sentences
    indices = np.random.randint(
        low=0, high=num_sentences, size=(num_samples, sample_size)
    )

    baseline_better: int = 0
    new_better: int = 0
    num_equal: int = 0
    for index in indices:
        baseline_bleu = calc_bleu_from_stats(baseline_stats.iloc[index]).score
        new_bleu = calc_bleu_from_stats(new_stats.iloc[index]).score
        if new_bleu > baseline_bleu:
            new_better += 1
        elif baseline_bleu > new_bleu:
            baseline_better += 1
        else:
            # If the baseline corpus and new corpus are identical, this
            # degenerate case may occur.
            num_equal += 1

    return PairedBootstrapOutput(
        baseline_bleu=calc_bleu_from_stats(baseline_stats),
        new_bleu=calc_bleu_from_stats(new_stats),
        num_samples=num_samples,
        baseline_better=baseline_better,
        num_equal=num_equal,
        new_better=new_better,
    )


def paired_bootstrap_resample_from_files(
    reference_file: str,
    baseline_file: str,
    new_file: str,
    num_samples: int = 1000,
    sample_size: Optional[int] = None,
) -> PairedBootstrapOutput:
    with open(reference_file, "r") as f:
        references: List[str] = [line for line in f]

    with open(baseline_file, "r") as f:
        baseline_translations: List[str] = [line for line in f]
    baseline_stats: pd.DataFrame = get_sufficient_stats(
        translations=baseline_translations, references=references
    )

    with open(new_file, "r") as f:
        new_translations: List[str] = [line for line in f]
    new_stats: pd.DataFrame = get_sufficient_stats(
        translations=new_translations, references=references
    )

    return paired_bootstrap_resample(
        baseline_stats=baseline_stats,
        new_stats=new_stats,
        num_samples=num_samples,
        sample_size=sample_size,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-file",
        type=str,
        required=True,
        help="Text file containing reference tokenized (with whitespace separator) sentences.",
    )
    parser.add_argument(
        "--baseline-file",
        type=str,
        required=True,
        help="Text file containing tokenized sentences translated by baseline system.",
    )
    parser.add_argument(
        "--new-file",
        type=str,
        required=True,
        help="Text file containing tokenized sentences translated by new system.",
    )
    args = parser.parse_args()

    output = paired_bootstrap_resample_from_files(
        reference_file=args.reference_file,
        baseline_file=args.baseline_file,
        new_file=args.new_file,
    )

    print(f"Baseline BLEU: {output.baseline_bleu.score:.2f}")
    print(f"New BLEU: {output.new_bleu.score:.2f}")
    print(f"BLEU delta: {output.new_bleu.score - output.baseline_bleu.score:.2f} ")
    print(
        f"Baseline better confidence: {output.baseline_better / output.num_samples:.2%}"
    )
    print(f"New better confidence: {output.new_better / output.num_samples:.2%}")


if __name__ == "__main__":
    main()
