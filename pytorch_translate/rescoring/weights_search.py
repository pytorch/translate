#!/usr/bin/env python3

import argparse
import pickle

import numpy as np
import torch
from fairseq.scoring import bleu
from pytorch_translate import vocab_constants
from pytorch_translate.data.dictionary import Dictionary
from pytorch_translate.generate import smoothed_sentence_bleu


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description=("Rescore generated hypotheses with extra models")
    )
    parser.add_argument(
        "--scores-info-export-path", type=str, help="Model scores for weights search"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1000,
        help="Number of iterations of random search",
    )
    parser.add_argument("--report-oracle-bleu", default=False, action="store_true")
    return parser


class DummyTask:
    """
    Default values for pad, eos, unk
    """

    def __init__(self):
        self.target_dictionary = Dictionary()


def evaluate_weights(scores_info, feature_weights, length_penalty):
    scorer = bleu.Scorer(
        bleu.BleuConfig(
            pad=vocab_constants.PAD_ID,
            eos=vocab_constants.EOS_ID,
            unk=vocab_constants.UNK_ID,
        )
    )

    for example in scores_info:
        weighted_scores = (example["scores"] * feature_weights).sum(axis=1)
        weighted_scores /= (example["tgt_len"] ** length_penalty) + 1e-12
        top_hypo_ind = np.argmax(weighted_scores)
        top_hypo = example["hypos"][top_hypo_ind]
        ref = example["target_tokens"]
        scorer.add(torch.IntTensor(ref), torch.IntTensor(top_hypo))

    return scorer.score()


def identify_nonzero_features(scores_info):
    nonzero_features = np.any(scores_info[0]["scores"] != 0, axis=0)
    for example in scores_info[1:]:
        nonzero_features |= np.any(example["scores"] != 0, axis=0)

    return np.where(nonzero_features)[0]


def random_search(scores_info_export_path, num_trials, report_oracle_bleu=False):
    with open(scores_info_export_path, "rb") as f:
        scores_info = pickle.load(f)

    dummy_task = DummyTask()

    if report_oracle_bleu:
        oracle_scorer = bleu.Scorer(
            bleu.BleuConfig(
                pad=vocab_constants.PAD_ID,
                eos=vocab_constants.EOS_ID,
                unk=vocab_constants.UNK_ID,
            )
        )

        for example in scores_info:
            smoothed_bleu = []
            for hypo in example["hypos"]:
                eval_score = smoothed_sentence_bleu(
                    dummy_task,
                    torch.IntTensor(example["target_tokens"]),
                    torch.IntTensor(hypo),
                )
                smoothed_bleu.append(eval_score)
            best_hypo_ind = np.argmax(smoothed_bleu)
            example["best_hypo_ind"] = best_hypo_ind

            oracle_scorer.add(
                torch.IntTensor(example["target_tokens"]),
                torch.IntTensor(example["hypos"][best_hypo_ind]),
            )

        print("oracle BLEU: ", oracle_scorer.score())

    num_features = scores_info[0]["scores"].shape[1]
    assert all(
        example["scores"].shape[1] == num_features for example in scores_info
    ), "All examples must have the same number of scores!"
    feature_weights = np.zeros(num_features)
    feature_weights[0] = 1
    score = evaluate_weights(scores_info, feature_weights, length_penalty=1)
    print("base BLEU: ", score)
    best_score = score
    best_weights = feature_weights
    best_length_penalty = 0

    nonzero_features = identify_nonzero_features(scores_info)

    for i in range(num_trials):
        feature_weights = np.zeros(num_features)
        random_weights = np.random.dirichlet(np.ones(nonzero_features.size))
        feature_weights[nonzero_features] = random_weights
        length_penalty = 1.5 * np.random.random()

        score = evaluate_weights(scores_info, feature_weights, length_penalty)
        if score > best_score:
            best_score = score
            best_weights = feature_weights
            best_length_penalty = length_penalty

        print(f"\r[{i}]  best: {best_score}", end="", flush=True)

    print()
    print("best weights: ", best_weights)
    print("best length penalty: ", length_penalty)

    return best_weights, best_length_penalty, best_score


def main():
    args = get_arg_parser().parse_args()

    assert (
        args.scores_info_export_path is not None
    ), "--scores-info-export-path is required for weights search"

    random_search(
        args.scores_info_export_path, args.num_trials, args.report_oracle_bleu
    )


if __name__ == "__main__":
    main()
