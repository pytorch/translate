#!/usr/bin/env python3

from collections import defaultdict
from typing import Dict

from pytorch_translate.research.unsupervised_morphology.ibm_model1 import IBMModel1
from pytorch_translate.research.unsupervised_morphology.unsupervised_morphology import (
    MorphologyHMMParams,
)


class BilingualMorphologyHMMParams(MorphologyHMMParams):
    def __init__(self, smoothing_const: float = 0.1, len_cost_pow: float = 2):
        """
        This class contains unigram HMM probabilities for the morphological model.
        Args:
            * smoothing_const: For smoothing the categorical distribution. This is
            mostly useful for unseen observations outside training.
            * len_cost_pow: used for penalizing long char sequences. Here we use
                it in emission as exp(- math.pow(len(str)-1, len_cost_pow))
        """
        super().__init__(smoothing_const, len_cost_pow)

        # For every word in the target language, we create a dictionary that gives
        # us information about the source language words that could be aligned to
        # them with their corresponding probabilities. This comes from IBM model
        # and is fixed during training.
        # This is actually p(target|source) but since it is easier to iterate
        # over target words first, we save it in reverse order.
        self.alignment_probs: Dict[str, Dict] = defaultdict()

    def init_params_from_data(
        self,
        src_path: str,
        dst_path: str,
        num_ibm_iters: int,
        num_candidate_per_word: int,
    ):
        """
        Args:
            num_candidate_per_word: Number of top-k words in the target language
                as translation candidates.
        """
        super().init_params_from_data(input_file_path=src_path)
        ibm_model = IBMModel1()
        ibm_model.learn_ibm_parameters(
            src_path=src_path, dst_path=dst_path, num_iters=num_ibm_iters
        )

        self.alignment_probs: Dict[str, Dict] = defaultdict()
        for src_word in ibm_model.translation_prob.keys():
            topk_pairs = sorted(
                ibm_model.translation_prob[src_word].items(), key=lambda x: x[1]
            )[:num_candidate_per_word]
            denom = sum(p[1] for p in topk_pairs)
            for candidate in topk_pairs:
                dst_word, prob = candidate[0], candidate[1] / denom
                if dst_word not in self.alignment_probs:
                    self.alignment_probs[dst_word] = defaultdict(float)
                self.alignment_probs[dst_word][src_word] = prob
