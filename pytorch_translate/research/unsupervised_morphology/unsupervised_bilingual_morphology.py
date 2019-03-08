#!/usr/bin/env python3

import copy
import pickle
from collections import defaultdict
from typing import Dict

from pytorch_translate.research.unsupervised_morphology import unsupervised_morphology


class BilingualMorphologyHMMParams(unsupervised_morphology.MorphologyHMMParams):
    def __init__(
        self,
        smoothing_const: float = 0.1,
        max_morph_len: int = 8,
        len_cost_pow: float = 2,
    ):
        """
        This class contains unigram HMM probabilities plus translation to the
            target language morphology.
        Args:
            * smoothing_const: For smoothing the categorical distribution. This is
            mostly useful for unseen observations outside training.
            * max_morph_len: maximum allowed size of a morpheme.
            * len_cost_pow: used for penalizing long char sequences. Here we use
                it in emission as exp(- math.pow(len(str)-1, len_cost_pow))
        """
        super().__init__(smoothing_const, max_morph_len, len_cost_pow)

        # Probability of translating morpheme x in source to different possible
        # values y in target: t(y|x)
        self.translation_probs: Dict[str, Dict] = defaultdict()

    def init_params_from_data(self, source_txt_file, target_txt_file):
        """
        We should obtain a list of all possible morphemes from parallel data.
        Args:
            source_txt_file: Source text file with the same number of line as
                target_txt_file.
            target_txt_file: Target text file of the parallel data.
        """
        super().init_params_from_data(input_file_path=source_txt_file)

        target_morphs = set()
        with open(target_txt_file, "r", encoding="utf-8") as input_stream:
            for line in input_stream:
                sen = line.strip()
                for i in range(0, len(sen)):
                    for j in range(i, min(i + self.max_morph_len, len(sen))):
                        target_morphs.add(sen[i : j + 1])

        # We just initialize the translation probabilities uniformly.
        self.translation_probs: Dict[str, Dict] = defaultdict()
        uniform_target_proabibility = {
            morph: 1.0 / len(self.morph_emit_probs) for morph in target_morphs
        }
        for source_morph in self.morph_emit_probs.keys():
            self.translation_probs[source_morph] = copy.deepcopy(
                uniform_target_proabibility
            )

    def zero_out_parmas(self):
        """
        Resets parameter values for all parameters.
        """
        super().zero_out_parmas()
        for morpheme in self.translation_probs.keys():
            for target_morpheme in self.translation_probs[morpheme].keys():
                self.translation_probs[morpheme][target_morpheme] = 0.0

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            e, s, lc, mml, tp = pickle.load(f)
        m = BilingualMorphologyHMMParams(s)
        m.morph_emit_probs = e
        m.len_cost_pow = lc
        m.max_morph_len = mml
        m.translation_probs = tp
        return m

    def save(self, file_path):
        e, s, lc, mml, tp = (
            self.morph_emit_probs,
            self.smoothing_const,
            self.len_cost_pow,
            self.max_morph_len,
            self.translation_probs,
        )
        with open(file_path, "wb") as f:
            pickle.dump((e, s, lc, mml, tp), f)
