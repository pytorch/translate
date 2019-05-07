#!/usr/bin/env python3

import argparse
import pickle
from enum import Enum

import torch
from fairseq import bleu
from pytorch_translate import hybrid_transformer_rnn  # noqa
from pytorch_translate.rescoring.model_scorers import (
    LMScorer,
    R2LModelScorer,
    ReverseModelScorer,
    SimpleModelScorer,
)
from pytorch_translate.research.rescore import cloze_transformer_model  # noqa
from pytorch_translate.tasks import pytorch_translate_task  # noqa
from tqdm import tqdm


class FeatureList(Enum):
    L2R_MODEL_SCORE = 0
    R2L_MODEL_SCORE = 1
    REVERSE_MODEL_SCORE = 2
    LM_SCORE = 3
    CLOZE_SCORE = 4


class Rescorer:
    """Reranks n-best hypotheses based on extra models and parameters"""

    def __init__(self, args):
        self.args = args

        assert (
            args.l2r_model_path is not None
        ), "Rescoring needs --l2r-model-path which generated given hypotheses"
        self.l2r_model_scorer = SimpleModelScorer(args, args.l2r_model_path)
        self.forward_task = self.l2r_model_scorer.task

        self.r2l_model_scorer = None
        if args.r2l_model_path:
            self.r2l_model_scorer = R2LModelScorer(args, args.r2l_model_path)

        self.reverse_model_scorer = None
        if args.reverse_model_path:
            self.reverse_model_scorer = ReverseModelScorer(
                args, args.reverse_model_path, self.forward_task
            )

        self.lm_scorer = None
        if args.lm_model_path:
            self.lm_scorer = LMScorer(args, args.lm_model_path, self.forward_task)

        self.cloze_transformer_scorer = None
        if args.cloze_transformer_path:
            self.cloze_transformer_scorer = SimpleModelScorer(
                args, args.cloze_transformer_path
            )

    def score(self, src_tokens, hypos):
        """run models and compute scores based on p(y), p(x|y) etc."""
        scores = torch.zeros((len(hypos), len(FeatureList)), dtype=torch.float)

        self.compute_l2r_model_scores(src_tokens, hypos, scores)
        self.compute_r2l_model_scores(src_tokens, hypos, scores)
        self.compute_reverse_model_scores(src_tokens, hypos, scores)
        self.compute_lm_scores(src_tokens, hypos, scores)
        self.compute_cloze_transformer_scores(src_tokens, hypos, scores)

        return scores

    def compute_l2r_model_scores(self, src_tokens, hypos, scores):
        l2r_scores = self.l2r_model_scorer.score(src_tokens, hypos)
        scores[:, FeatureList.L2R_MODEL_SCORE.value] = l2r_scores[:]

    def compute_r2l_model_scores(self, src_tokens, hypos, scores):
        if not self.r2l_model_scorer:
            return
        r2l_scores = self.r2l_model_scorer.score(src_tokens, hypos)
        scores[:, FeatureList.R2L_MODEL_SCORE.value] = r2l_scores[:]

    def compute_reverse_model_scores(self, src_tokens, hypos, scores):
        """computes p(x|y) for each hypothesis. """
        if not self.reverse_model_scorer:
            return

        scores[
            :, FeatureList.REVERSE_MODEL_SCORE.value
        ] = self.reverse_model_scorer.score(src_tokens, hypos)

    def compute_lm_scores(self, src_tokens, hypos, scores):
        """computes p(x|y) for each hypothesis. """
        if not self.lm_scorer:
            return

        lm_scores = self.lm_scorer.score(src_tokens, hypos)
        scores[:, FeatureList.LM_SCORE.value] = lm_scores[:]

    def compute_cloze_transformer_scores(self, src_tokens, hypos, scores):
        if not self.cloze_transformer_scorer:
            return

        cloze_scores = self.cloze_transformer_scorer.score(src_tokens, hypos)
        scores[:, FeatureList.CLOZE_SCORE.value] = cloze_scores[:]


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description=("Rescore generated hypotheses with extra models")
    )
    parser.add_argument(
        "--translation-info-export-path",
        default=None,
        type=str,
        help=("Optional path to save translation info output in pickled format"),
    )
    parser.add_argument(
        "--l2r-model-path",
        default=None,
        type=str,
        help=("Provide a path for the l2r rescoring model"),
    )
    parser.add_argument(
        "--l2r-model-weight",
        default=1.0,
        type=float,
        help=("Provide a weight for the l2r rescoring model"),
    )
    parser.add_argument(
        "--r2l-model-path",
        default=None,
        type=str,
        help=("Provide a path for the r2l rescoring model"),
    )
    parser.add_argument(
        "--r2l-model-weight",
        default=1.0,
        type=float,
        help=("Provide a weight for the r2l rescoring model"),
    )
    parser.add_argument(
        "--reverse-model-path",
        default=None,
        type=str,
        help=("Provide a path for the reverse rescoring model"),
    )
    parser.add_argument(
        "--reverse-model-weight",
        default=1.0,
        type=float,
        help=("Provide a weight for the reverse rescoring model"),
    )
    parser.add_argument(
        "--lm-model-path",
        default=None,
        type=str,
        help=("Provide a path for the language model rescoring model"),
    )
    parser.add_argument(
        "--lm-model-weight",
        default=1.0,
        type=float,
        help=("Provide a weight for the lm rescoring model"),
    )
    parser.add_argument(
        "--length-penalty",
        default=1.0,
        type=float,
        help=("Provide a weight for length penalty used in rescoring"),
    )
    parser.add_argument(
        "--cloze-transformer-path",
        default=None,
        type=str,
        help=("Provide a path for the cloze transformer rescoring model"),
    )
    parser.add_argument(
        "--cloze-transformer-weight",
        default=1.0,
        type=float,
        help=("Provide a weight for the cloze transformer model"),
    )
    return parser


def combine_weighted_scores(scores, weights, src_len, tgt_len, lenpen):
    """ Combines scores from different models and returns

    Args:
        scores: scores for each feature and hypo [num_of_hypos, num_of_features]
        weights: weights for each feature [num_of_features]
        src_len: number of source sentence tokens
        tgt_len: list of target sentence tokens lengths [num_of_hypos]
        lenpen: float representing length penalty

    Returns:
        weighted_scores: one unified score for each hypothesis [num_of_hypos]
    """
    weighted_scores = scores.clone()
    weighted_scores[:, FeatureList.L2R_MODEL_SCORE.value] /= tgt_len ** lenpen
    weighted_scores[:, FeatureList.R2L_MODEL_SCORE.value] /= tgt_len ** lenpen
    weighted_scores[:, FeatureList.REVERSE_MODEL_SCORE.value] /= src_len ** lenpen
    weighted_scores[:, FeatureList.LM_SCORE.value] /= tgt_len ** lenpen
    weighted_scores[:, FeatureList.CLOZE_SCORE.value] /= tgt_len ** lenpen

    weighted_scores *= torch.tensor(weights)
    # convert [num_of_hypos, num_of_features] to [num_of_hypos] and return
    return weighted_scores.sum(dim=1)


def find_top_tokens(args, trans_info, rescorer):
    """ Rescore translations and combine weights to find top hypo tokens
    """
    src_tokens = trans_info["src_tokens"].cuda()
    hypos = trans_info["hypos"]

    scores = rescorer.score(src_tokens, hypos)

    # Prepare all the weights and call combine weighted scores
    weights = [
        args.l2r_model_weight,
        args.r2l_model_weight,
        args.reverse_model_weight,
        args.lm_model_weight,
        args.cloze_transformer_weight,
    ]
    src_len = len(src_tokens)
    tgt_len = torch.tensor([len(hypo["tokens"]) for hypo in hypos], dtype=torch.float)
    combined_scores = combine_weighted_scores(
        scores, weights, src_len, tgt_len, args.length_penalty
    )
    top_index = combined_scores.max(0)[1]
    return hypos[top_index]["tokens"]


def main():
    args = get_arg_parser().parse_args()

    assert (
        args.translation_info_export_path is not None
    ), "--translation_info_export_path is required for rescoring"

    rescorer = Rescorer(args)
    dst_dict = rescorer.forward_task.tgt_dict
    base_bleu_scorer = bleu.Scorer(dst_dict.pad(), dst_dict.eos(), dst_dict.unk())
    rescoring_bleu_scorer = bleu.Scorer(dst_dict.pad(), dst_dict.eos(), dst_dict.unk())

    translation_info_list = pickle.load(open(args.translation_info_export_path, "rb"))
    for trans_info in tqdm(translation_info_list):
        trans_info["hypos"] = [
            {"score": hypo["score"], "tokens": hypo["tokens"].cuda()}
            for hypo in trans_info["hypos"]
        ]

        base_bleu_scorer.add(
            trans_info["target_tokens"].int().cpu(),
            trans_info["hypos"][0]["tokens"].int().cpu(),
        )

        top_tokens = find_top_tokens(args, trans_info, rescorer)

        rescoring_bleu_scorer.add(
            trans_info["target_tokens"].int().cpu(), top_tokens.int().cpu()
        )

    print("| Base ", base_bleu_scorer.result_string())
    print("| Rescoring ", rescoring_bleu_scorer.result_string())


if __name__ == "__main__":
    main()
