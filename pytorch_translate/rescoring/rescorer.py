from __future__ import absolute_import, division, print_function, unicode_literals

from enum import Enum

import torch
from pytorch_translate.rescoring.model_scorers import (
    LMScorer,
    R2LModelScorer,
    ReverseModelScorer,
    SimpleModelScorer,
)


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
            self.args.word_reward == 0.0 and self.args.length_penalty != 0.0
        ), "For rescoring, L2R model should be scored with length penalty"

        assert (
            args.l2r_model_path is not None
        ), "Rescoring needs --l2r-model-path which generated given hypotheses"
        self.l2r_model_scorer = SimpleModelScorer(args, args.l2r_model_path)
        self.forward_task = self.l2r_model_scorer.task

        if args.enable_r2l_rescoring:
            assert (
                args.r2l_model_path
            ), "Provide --r2l-model-path with --enable_r2l_scoring"
            self.r2l_model_scorer = R2LModelScorer(args, args.r2l_model_path)

        if args.enable_reverse_rescoring:
            assert (
                args.reverse_model_path
            ), "Provide --reverse-model-path with --enable-reverse-scoring"
            self.reverse_model_scorer = ReverseModelScorer(
                args, args.reverse_model_path, self.forward_task
            )

        if args.enable_lm_rescoring:
            assert (
                args.lm_model_path
            ), "Provide --lm-model-path with --enable-lm-scoring"
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
        if not self.args.enable_r2l_rescoring:
            return
        r2l_scores = self.r2l_model_scorer.score(src_tokens, hypos)
        scores[:, FeatureList.R2L_MODEL_SCORE.value] = r2l_scores[:]

    def compute_reverse_model_scores(self, src_tokens, hypos, scores):
        """computes p(x|y) for each hypothesis. """
        if not self.args.enable_reverse_rescoring:
            return

        scores[
            :, FeatureList.REVERSE_MODEL_SCORE.value
        ] = self.reverse_model_scorer.score(src_tokens, hypos)

    def compute_lm_scores(self, src_tokens, hypos, scores):
        """computes p(x|y) for each hypothesis. """
        if not self.args.enable_lm_rescoring:
            return

        lm_scores = self.lm_scorer.score(src_tokens, hypos)
        scores[:, FeatureList.LM_SCORE.value] = lm_scores[:]
