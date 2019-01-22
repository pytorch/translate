#!/usr/bin/env python3

import math
from collections import Counter


class MorphologyHMMParams(object):
    def __init__(self, smoothing_const=0.1):
        """
        This class contains HMM probabilities for the morphological model.
        The model has transition probabilities of affix classes, and
        emission probabilities of morphemes.
        Types of affixes: prefix, stem, suffix (we ignore infixes).
        Example of transition probabilities: q(suffix | stem).
        Example of emission probabilities: e(ing | suffix).
        We also have two registered affixes for showing the START and END of
        words.
        Args:
            smoothing_const: for smoothing the categorical distribution. This is
            mostly useful for unseen observations outside training.
        """
        self.morph_emit_probs = {"prefix": {}, "stem": {}, "suffix": {}}
        self.affix_trans_probs = {
            "prefix": {},
            "stem": {},
            "suffix": {},
            "START": {},
            "END": {},
        }
        self.word_counts = Counter()
        self.smoothing_const = smoothing_const
        self.SMALL_CONST = -10000

    def init_params_from_data(self, input_file_path):
        """
        We should obtain a list of all possible morphemes from a data file.
        """
        self.morph_emit_probs = {"prefix": {}, "stem": {}, "suffix": {}}
        self.affix_trans_probs = {
            "prefix": {},
            "stem": {},
            "suffix": {},
            "START": {},
            "END": {},
        }

        with open(input_file_path, "r", encoding="utf-8") as input_stream:
            for line in input_stream:
                for word in line.strip().split():
                    self.word_counts[word] += 1

        for word in self.word_counts:
            if len(word) <= 2:
                # If a word is very short, it is definitely an
                # independent stem.
                self.morph_emit_probs["stem"][word] = 1.0

            for i in range(0, len(word)):
                for j in range(i, len(word)):
                    if j < len(word) - 2:
                        self.morph_emit_probs["prefix"][word[i : j + 1]] = 1.0
                    if j - i >= 1:
                        self.morph_emit_probs["stem"][word[i : j + 1]] = 1.0
                    if i >= 2:
                        self.morph_emit_probs["suffix"][word[i : j + 1]] = 1.0

        # Normalizing the initial probabilities uniformly.
        for affix in self.morph_emit_probs.keys():
            num_morphs = len(self.morph_emit_probs[affix])
            for morph in self.morph_emit_probs[affix].keys():
                self.morph_emit_probs[affix][morph] = (1.0 + self.smoothing_const) / (
                    num_morphs * (1 + self.smoothing_const)
                )

        # Initializing the transition probabilities. Here we force some trivial
        # rules such as not having a suffix right after a prefix.
        self.affix_trans_probs["START"]["START"] = 0
        self.affix_trans_probs["START"]["prefix"] = 0.5
        self.affix_trans_probs["START"]["stem"] = 0.5
        self.affix_trans_probs["START"]["suffix"] = 0
        self.affix_trans_probs["START"]["END"] = 0
        self.affix_trans_probs["prefix"]["START"] = 0
        self.affix_trans_probs["prefix"]["prefix"] = 0.5
        self.affix_trans_probs["prefix"]["stem"] = 0.5
        self.affix_trans_probs["prefix"]["suffix"] = 0
        self.affix_trans_probs["prefix"]["END"] = 0
        self.affix_trans_probs["stem"]["START"] = 0
        self.affix_trans_probs["stem"]["prefix"] = 0
        self.affix_trans_probs["stem"]["stem"] = 1.0 / 3
        self.affix_trans_probs["stem"]["suffix"] = 1.0 / 3
        self.affix_trans_probs["stem"]["END"] = 1.0 / 3
        self.affix_trans_probs["suffix"]["START"] = 0
        self.affix_trans_probs["suffix"]["prefix"] = 0
        self.affix_trans_probs["suffix"]["stem"] = 0
        self.affix_trans_probs["suffix"]["suffix"] = 0.5
        self.affix_trans_probs["suffix"]["END"] = 0.5
        self.affix_trans_probs["END"]["START"] = 0
        self.affix_trans_probs["END"]["prefix"] = 0
        self.affix_trans_probs["END"]["stem"] = 0
        self.affix_trans_probs["END"]["suffix"] = 0
        self.affix_trans_probs["END"]["END"] = 0

    def transition_log_prob(self, prev_affix, current_affix):
        """
        For parameter estimation of the forward-backward algorithm, we still need
        the actual probability values. However, in decoding, working in log-scale
        is more efficient.
        """
        if self.affix_trans_probs[prev_affix][current_affix] == 0:
            return self.SMALL_CONST
        return math.log(self.affix_trans_probs[prev_affix][current_affix])

    def emission_prob(self, affix, morpheme):
        """
        If a morpheme is not seen, we return the default smoothed probability.
        Note that we assume the probabilities for the seen morphemes are already
        smoothed.
        """
        if affix in {"START", "END"}:
            return 0
        if morpheme in self.morph_emit_probs[affix]:
            return self.morph_emit_probs[affix][morpheme]
        else:
            num_morphs = len(self.morph_emit_probs[affix])
            return self.smoothing_const / (num_morphs * (1 + self.smoothing_const))

    def emission_log_probs(self, affix, morpheme):
        if affix in {"START", "END"}:
            return self.SMALL_CONST
        return math.log(self.emission_prob(affix, morpheme))
