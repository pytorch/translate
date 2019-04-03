from __future__ import absolute_import, division, print_function, unicode_literals

from enum import Enum

import torch
from pytorch_translate.rescoring.model_scorers import (
    LMScorer,
    R2LModelScorer,
    ReverseModelScorer,
)


class FeatureList(Enum):
    ORIGINAL_MODEL_SCORE = 0
    R2L_MODEL_SCORE = 1
    REVERSE_MODEL_SCORE = 2
    LM_SCORE = 3


class Rescorer:
    """Reranks n-best hypotheses based on extra models and parameters"""

    def __init__(self, args, original_task):
        self.args = args
        self.original_task = original_task

        if args.enable_r2l_rescoring:
            assert (
                args.r2l_model_path
            ), "Provide --r2l-model-path with --enable_r2l_scoring"
            self.r2l_model_scorer = R2LModelScorer(
                args, args.r2l_model_path, self.original_task
            )

        if args.enable_reverse_rescoring:
            assert (
                args.reverse_model_path
            ), "Provide --reverse-model-path with --enable-reverse-scoring"
            self.reverse_model_scorer = ReverseModelScorer(
                args, args.reverse_model_path, self.original_task
            )

        if args.enable_lm_rescoring:
            assert (
                args.lm_model_path
            ), "Provide --lm-model-path with --enable-lm-scoring"
            self.lm_scorer = LMScorer(args, args.lm_model_path, self.original_task)

    def combine_weighted_scores(self, scores, src_tokens, hypos):
        """combine scores from different models"""
        src_len = torch.tensor(len(src_tokens), dtype=torch.float)
        tgt_len = torch.tensor(
            [len(hypo["tokens"]) for hypo in hypos], dtype=torch.float
        )

        scores[:, FeatureList.ORIGINAL_MODEL_SCORE.value] *= (
            self.args.original_model_weight / tgt_len
        )
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

        self.compute_original_model_scores(src_tokens, hypos, scores)
        self.compute_r2l_model_scores(src_tokens, hypos, scores)
        self.compute_reverse_model_scores(src_tokens, hypos, scores)
        self.compute_lm_scores(src_tokens, hypos, scores)

        max_score_index = self.combine_weighted_scores(scores, src_tokens, hypos)
        return hypos[max_score_index]["tokens"].int().cpu()

    def compute_original_model_scores(self, src_tokens, hypos, scores):
        for i, hypo in enumerate(hypos):
            scores[i, FeatureList.ORIGINAL_MODEL_SCORE.value] = hypo["score"]

    def compute_r2l_model_scores(self, src_tokens, hypos, scores):
        if not self.args.enable_r2l_rescoring:
            return
        r2l_scores = self.r2l_model_scorer.score(src_tokens, hypos)
        scores[:, FeatureList.R2L_MODEL_SCORE.value] = r2l_scores[:]

    def compute_reverse_model_scores(self, src_tokens, hypos, scores):
        """computes p(x|y) for each hypothesis. """
        if not self.args.enable_reverse_rescoring:
            return

        for i, hypo in enumerate(hypos):
            scores[
                i, FeatureList.REVERSE_MODEL_SCORE.value
            ] = self.reverse_model_scorer.score(src_tokens, hypo)

    def compute_lm_scores(self, src_tokens, hypos, scores):
        """computes p(x|y) for each hypothesis. """
        if not self.args.enable_lm_rescoring:
            return

        lm_scores = self.lm_scorer.score(src_tokens, hypos)
        scores[:, FeatureList.LM_SCORE.value] = lm_scores[:]
