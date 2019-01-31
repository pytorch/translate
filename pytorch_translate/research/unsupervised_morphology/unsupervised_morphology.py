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
        if self.emission_prob(affix, morpheme) == 0:
            return self.SMALL_CONST
        return math.log(self.emission_prob(affix, morpheme))


class MorphologySegmentor(object):
    def __init__(self, morphology_hmm_params):
        self.params = morphology_hmm_params

    def initialize_prob_values(self, n):
        pi = [{} for _ in range(n + 1)]
        pi[0]["START"] = 0
        pi[0]["prefix"] = self.params.SMALL_CONST
        pi[0]["stem"] = self.params.SMALL_CONST
        pi[0]["suffix"] = self.params.SMALL_CONST
        pi[0]["END"] = self.params.SMALL_CONST
        for i in range(1, n + 1):
            for affix in self.params.affix_trans_probs.keys():
                pi[i][affix] = (i + 1) * self.params.SMALL_CONST
        return pi

    def segment_viterbi(self, word):
        """
        This is a dynamic programming algorithm for segmenting a word by using a
        modified version of the Viterbi algorithm. The main difference here is that
        the segment-Viterbi algorithm is based on segment-HMM where a state can
        emit more than one output. In our case, a state is an affix such as prefix,
        and an output is a substring of a word. Because of searching for a substring
        the segment-Viterbi algorithm has a slower run-time. In this case (bigram HMM),
        the runtime complexity of a vanilla Viterbi algorithm is O(nT^2) where n
        is the number of characters in word and T is the number of possible affixes
        (in our case 3). The segment-Viterbi has a complexity O(n^2 T^2).
        A pseudo-code for the vanilla Viterbi algorithm:
        http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf#page=18
        """
        n = len(word)
        pi = self.initialize_prob_values(n)
        back_pointer = [{} for _ in range(n + 1)]

        for i in range(n):  # loop over starting indices of morphemes
            for j in range(i + 1, n + 1):  # loop over end indices of morphemes
                for v in {"prefix", "stem", "suffix"}:  # loop over possible tags
                    for w in {"START", "prefix", "stem", "suffix"}:
                        # loop over possible previous tags
                        t = self.params.transition_log_prob(w, v)
                        e = self.params.emission_log_probs(v, word[i:j])
                        log_prob = pi[i][w] + t + e
                        if log_prob > pi[j][v]:
                            pi[j][v] = log_prob
                            # Saving backpointer for previous tag and index.
                            back_pointer[j][v] = w, i

        # finalizing the best segmentation.
        best_score = n * self.params.SMALL_CONST
        best_bp = None
        indices = [n]  # backtracking indices for segmentation.
        labels = []  # backtracking labels for segmentation.
        for v in {"prefix", "stem", "suffix"}:
            t = self.params.transition_log_prob(v, "END")
            log_prob = pi[n][v] + t
            if log_prob > best_score:
                best_score = log_prob
                best_bp = back_pointer[n][v][1], v, back_pointer[n][v][0]

        indices.append(best_bp[0])
        labels.append(best_bp[1])
        if best_bp[0] > 0:
            # In cases where the last index is zero, we don't want to backtrack
            # anymore.
            labels.append(best_bp[2])

        while True:
            last_index = indices[-1]
            last_label = labels[-1]
            if last_index == 0:
                break
            w, i = back_pointer[last_index][last_label]
            indices.append(i)
            if i == 0:
                break
            labels.append(w)

        # We should now reverse the backtracked list.
        indices.reverse()
        labels.reverse()
        return labels, indices
