#!/usr/bin/env python3

import copy
import math
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

    def translation_prob(self, src_morph: str, target_morph: str):
        if (
            src_morph in self.translation_probs
            and target_morph in self.translation_probs[src_morph]
        ):
            return self.translation_probs[src_morph][target_morph]
        return 0.0

    def translation_log_prob(self, src_morph: str, target_morph: str):
        trans_prob = self.translation_prob(src_morph, target_morph)
        if trans_prob == 0.0:
            return self.SMALL_CONST
        return math.log(trans_prob)

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


class BilingualMorphologySegmentor(unsupervised_morphology.MorphologySegmentor):
    def segment_blingual_viterbi(self, src_sentence: str, dst_sentence: str):
        """
        This is a dynamic programming algorithm for segmenting a sentence by using
        a modified version of the Viterbi algorithm that also uses the IBM model.
        The main differences to MorphologySegmentor: it uses IBM model parameters
        to find the best segmentation.
        """
        src_len, dst_len = len(src_sentence), len(dst_sentence)
        pi = [self.params.SMALL_CONST for _ in range(src_len + 1)]
        pi[0] = 0
        back_pointer = [0 for _ in range(src_len + 1)]

        # We first calculate the relative count of each morpheme in the target
        # language.
        target_morpheme_counts: Dict[str, float] = defaultdict(float)
        for dst_start in range(dst_len):
            # loop over end indices of morphemes.
            for dst_end in range(
                dst_start + 1, min(dst_len + 1, dst_start + self.params.max_morph_len)
            ):
                substr = dst_sentence[dst_start:dst_end]
                target_morpheme_counts[substr] += 1
        denom = sum(target_morpheme_counts.values())

        target_morpheme_log_probs = {
            morpheme: math.log(target_morpheme_counts[morpheme] / denom)
            for morpheme in target_morpheme_counts.keys()
        }

        # loop over starting indices of morphemes.
        for src_start in range(src_len):
            # loop over end indices of morphemes.
            for src_end in range(
                src_start + 1, min(src_len + 1, src_start + self.params.max_morph_len)
            ):
                substr = src_sentence[src_start:src_end]
                e = self.params.emission_log_prob(substr)

                # Getting max prability with respect to the target side.
                max_target_prob = max(
                    target_morpheme_log_probs[morpheme]
                    + self.params.translation_log_prob(substr, morpheme)
                    for morpheme in target_morpheme_log_probs.keys()
                )

                log_prob = pi[src_start] + e + max_target_prob
                if log_prob > pi[src_end]:
                    pi[src_end] = log_prob
                    # Saving backpointer for previous tag and index.
                    back_pointer[src_end] = src_start

        # finalizing the best segmentation.
        indices = [src_len]  # backtracking indices for segmentation.
        indices.append(back_pointer[-1])
        while True:
            last_index = indices[-1]
            if last_index == 0:
                break
            start_index = back_pointer[last_index]
            indices.append(start_index)
            if start_index == 0:
                break

        # We should now reverse the backtracked list.
        indices.reverse()
        return indices
