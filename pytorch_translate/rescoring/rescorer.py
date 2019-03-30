from __future__ import absolute_import, division, print_function, unicode_literals

from enum import Enum

import numpy
from pytorch_translate.rescoring.model_scorers import R2LModelScorer, ReverseModelScorer


class FeatureList(Enum):
    ORIGINAL_MODEL_SCORE = 0
    R2L_MODEL_SCORE = 1
    REVERSE_MODEL_SCORE = 2


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

    def optimize(self, scores):
        """combine scores from different models"""
        return numpy.argmax(scores.sum(axis=1))

    def score(self, src_tokens, hypos):
        """run models and compute scores based on p(y), p(x|y) etc."""
        scores = numpy.zeros(shape=(len(hypos), len(FeatureList)))

        self.compute_r2l_model_scores(src_tokens, hypos, scores)
        self.compute_reverse_model_scores(src_tokens, hypos, scores)

        max_score_index = self.optimize(scores)
        return hypos[max_score_index]["tokens"].int().cpu()

    def compute_r2l_model_scores(self, src_tokens, hypos, scores):
        if not self.args.enable_r2l_rescoring:
            return
        r2l_scores = self.r2l_model_scorer.score(src_tokens, hypos).numpy()
        scores[:, FeatureList.R2L_MODEL_SCORE.value] = r2l_scores[:]

    def compute_reverse_model_scores(self, src_tokens, hypos, scores):
        """computes p(x|y) for each hypothesis. """
        if not self.args.enable_reverse_rescoring:
            return

        for i, hypo in enumerate(hypos):
            scores[
                i, FeatureList.REVERSE_MODEL_SCORE.value
            ] = self.reverse_model_scorer.score(src_tokens, hypo)
