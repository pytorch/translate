from __future__ import absolute_import, division, print_function, unicode_literals

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
from pytorch_translate.tasks import pytorch_translate_task  # noqa
from tqdm import tqdm


class FeatureList(Enum):
    L2R_MODEL_SCORE = 0
    R2L_MODEL_SCORE = 1
    REVERSE_MODEL_SCORE = 2
    LM_SCORE = 3


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

    def combine_weighted_scores(self, scores, src_tokens, hypos):
        """combine scores from different models"""
        src_len = torch.tensor(len(src_tokens), dtype=torch.float)
        tgt_len = torch.tensor(
            [len(hypo["tokens"]) for hypo in hypos], dtype=torch.float
        )

        scores[
            :, FeatureList.L2R_MODEL_SCORE.value
        ] *= (
            self.args.l2r_model_weight
        )  # L2R model score should be length normalized already
        scores[:, FeatureList.R2L_MODEL_SCORE.value] *= (
            self.args.r2l_model_weight / tgt_len
        )
        scores[:, FeatureList.REVERSE_MODEL_SCORE.value] *= (
            self.args.reverse_model_weight / src_len
        )
        scores[:, FeatureList.LM_SCORE.value] *= self.args.lm_model_weight / src_len
        return scores.sum(dim=1).max(0)[1]

    def score(self, src_tokens, hypos):
        """run models and compute scores based on p(y), p(x|y) etc."""
        scores = torch.zeros((len(hypos), len(FeatureList)), dtype=torch.float)

        self.compute_l2r_model_scores(src_tokens, hypos, scores)
        self.compute_r2l_model_scores(src_tokens, hypos, scores)
        self.compute_reverse_model_scores(src_tokens, hypos, scores)
        self.compute_lm_scores(src_tokens, hypos, scores)

        max_score_index = self.combine_weighted_scores(scores, src_tokens, hypos)
        return hypos[max_score_index]["tokens"].int().cpu()

    def compute_l2r_model_scores(self, src_tokens, hypos, scores):
        for i, hypo in enumerate(hypos):
            scores[i, FeatureList.L2R_MODEL_SCORE.value] = hypo["score"]

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
    return parser


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

        rescoring_top_tokens = rescorer.score(
            trans_info["src_tokens"].cuda(), trans_info["hypos"]
        )
        rescoring_bleu_scorer.add(
            trans_info["target_tokens"].int().cpu(), rescoring_top_tokens.int().cpu()
        )

    print("| Base ", base_bleu_scorer.result_string())
    print("| Rescoring ", rescoring_bleu_scorer.result_string())


if __name__ == "__main__":
    main()
