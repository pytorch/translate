from __future__ import absolute_import, division, print_function, unicode_literals

from enum import Enum

import numpy
from pytorch_translate.rescoring.model_scorers import R2LModelScorer


class FeatureList(Enum):
    ORIGINAL_MODEL_SCORE = 0
    R2L_MODEL_SCORE = 1


class Rescorer:
    """Reranks n-best hypotheses based on extra models and parameters"""

    def __init__(self, args):
        self.args = args

        if args.enable_r2l_rescoring:
            assert (
                args.r2l_model_path
            ), "You should provide r2l-model-path together with enable_r2l_scoring"
            self.r2l_model_scorer = R2LModelScorer(args, args.r2l_model_path)

    def optimize(self, scores):
        """combine scores from different models"""
        return numpy.argmax(scores.sum(axis=1))

    def score(self, src_tokens, hypos):
        """run models and compute scores based on p(y), p(x|y) etc."""
        scores = numpy.zeros(shape=(len(hypos), len(FeatureList)))

        self.compute_r2l_model_scores(src_tokens, hypos, scores)

        max_score_index = self.optimize(scores)
        return hypos[max_score_index]["tokens"].int().cpu()

    def compute_r2l_model_scores(self, src_tokens, hypos, scores):
        if not self.args.enable_r2l_rescoring:
            return
        r2l_scores = self.r2l_model_scorer.score(src_tokens, hypos).numpy()
        scores[:, FeatureList.R2L_MODEL_SCORE.value] = r2l_scores[:]
