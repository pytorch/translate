#!/usr/bin/env python3

from collections import defaultdict
from typing import Dict

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

        # For every word in the source language, we create a dictionary that gives
        # us information about the target language words that could be aligned to
        # them with their corresponding probabilities.
        self.alignment_probs: Dict[str, Dict] = defaultdict()

    def init_params_from_data(self, src_file_path):
        super().init_params_from_data(input_file_path=src_file_path)
        for word in self.word_counts.keys():
            self.alignment_probs[word] = defaultdict(float)
