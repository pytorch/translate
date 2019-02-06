#!/usr/bin/env python3

import math
import pickle
from collections import Counter, defaultdict
from itertools import chain, zip_longest
from multiprocessing import Pool


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

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            e, t, s, c = pickle.load(f)
        m = MorphologyHMMParams(s)
        m.morph_emit_probs = e
        m.affix_trans_probs = t
        m.word_counts = c
        return m

    def save(self, file_path):
        e, t, s, c = (
            self.morph_emit_probs,
            self.affix_trans_probs,
            self.smoothing_const,
            self.word_counts,
        )
        with open(file_path, "wb") as f:
            pickle.dump((e, t, s, c), f)


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

        for start in range(n):  # loop over starting indices of morphemes.
            for end in range(start + 1, n + 1):  # loop over end indices of morphemes.
                for cur_tag in ["prefix", "stem", "suffix"]:  # loop over possible tags.
                    for prev_tag in ["START", "prefix", "stem", "suffix"]:
                        # loop over possible previous tags
                        t = self.params.transition_log_prob(prev_tag, cur_tag)
                        e = self.params.emission_log_probs(cur_tag, word[start:end])
                        log_prob = pi[start][prev_tag] + t + e
                        if log_prob > pi[end][cur_tag]:
                            pi[end][cur_tag] = log_prob
                            # Saving backpointer for previous tag and index.
                            back_pointer[end][cur_tag] = prev_tag, start

        # finalizing the best segmentation.
        best_score = n * self.params.SMALL_CONST
        best_bp = None
        indices = [n]  # backtracking indices for segmentation.
        labels = []  # backtracking labels for segmentation.
        for last_tag in {"prefix", "stem", "suffix"}:
            t = self.params.transition_log_prob(last_tag, "END")
            log_prob = pi[n][last_tag] + t
            if log_prob > best_score:
                best_score = log_prob
                best_bp = (
                    back_pointer[n][last_tag][1],
                    last_tag,
                    back_pointer[n][last_tag][0],
                )

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
            prev_tag, start_index = back_pointer[last_index][last_label]
            indices.append(start_index)
            if start_index == 0:
                break
            labels.append(prev_tag)

        # We should now reverse the backtracked list.
        indices.reverse()
        labels.reverse()
        return labels, indices

    def segment_word(self, word, add_affix_symbols=False):
        """
        This method segments words based on the Viterbi algorithm. Given an input
        string, the algorithm segments that word to one or more substrings.

        Args:
            add_affix_symbols: if True, put a + after prefixes, and before affixes.
            Example: pretokenize --> pre+ token +ize
        """
        labels, indices = self.segment_viterbi(word)
        outputs = []
        for i in range(len(labels)):
            substr = word[indices[i] : indices[i + 1]]
            label = labels[i]
            if add_affix_symbols:
                if label == "prefix":
                    substr = substr + "+"
                elif label == "suffix":
                    substr = "+" + substr
            outputs.append(substr)

        return " ".join(outputs)


class UnsupervisedMorphology(object):
    def __init__(self, input_file, smoothing_const=0.1):
        self.params = MorphologyHMMParams(smoothing_const)
        self.params.init_params_from_data(input_file)

    @staticmethod
    def init_forward_values(n):
        forward_values = [{} for _ in range(n + 1)]
        forward_values[0]["START"] = 1.0
        forward_values[0]["prefix"] = 0
        forward_values[0]["stem"] = 0
        forward_values[0]["suffix"] = 0
        forward_values[0]["END"] = 0
        for i in range(1, len(forward_values)):
            for cur_tag in ["START", "prefix", "stem", "suffix", "END"]:
                forward_values[i][cur_tag] = 0
        return forward_values

    @staticmethod
    def init_backward_values(n):
        backward_values = [{} for _ in range(n + 1)]

        backward_values[n]["END"] = 1.0

        for i in range(0, len(backward_values)):
            for cur_tag in ["START", "prefix", "stem", "suffix", "END"]:
                backward_values[i][cur_tag] = 0
        return backward_values

    def forward(self, word):
        """
        For the forward pass in the forward-backward algorithm.
        The forward pass is very similar to the Viterbi algorithm. The main difference
        is that here we use summation instead of argmax.
        """
        n = len(word)
        forward_values = UnsupervisedMorphology.init_forward_values(n)
        for start in range(n):
            for end in range(start + 1, n + 1):
                for cur_tag in ["prefix", "stem", "suffix"]:
                    for prev_tag in ["START", "prefix", "stem", "suffix"]:
                        t = self.params.affix_trans_probs[prev_tag][cur_tag]
                        e = self.params.emission_prob(cur_tag, word[start:end])
                        forward_values[end][cur_tag] += (
                            forward_values[start][prev_tag] * t * e
                        )
        return forward_values

    def backward(self, word):
        """
        For the backward pass in the forward-backward algorithm.
        """
        n = len(word)
        backward_values = UnsupervisedMorphology.init_backward_values(n)

        # Here we initialize the backward pass of the valid morpheme classes equal
        # to their transition to the END state.
        for cur_tag in ["prefix", "stem", "suffix"]:
            t = self.params.affix_trans_probs[cur_tag]["END"]
            backward_values[n][cur_tag] = t

        # Here we do regular summation over all possible paths leading to a specific
        # morpheme class.
        for start in range(n - 1, -1, -1):
            for end in range(start + 1, n + 1):
                for cur_tag in ["prefix", "stem", "suffix"]:
                    for next_tag in ["prefix", "stem", "suffix"]:
                        t = self.params.affix_trans_probs[cur_tag][next_tag]
                        e = self.params.emission_prob(next_tag, word[start:end])
                        backward_values[start][cur_tag] += (
                            backward_values[end][next_tag] * t * e
                        )
        return backward_values

    def forward_backward(self, word):
        """
        The forward-backward algorithm is a dynamic programming algorithm that
        enumerates all possible observations in a sequence and returns the
        probability of the every parameter. These probabilities can be used in
        the expectation-maximization (EM) algorithm for estimating the expected
        values for each observation in the data. Here is the implementation of the
        FB algorithm for the segment HMM model. This algorithm, especially the
        forward pass, is analogous to the Viterbi algorithm: the FB algorithm is
        an instance of the sum-product algorithm while the Viterbi algorithm is
        an instance of the max-product algorithm.
        A pseudo-code of forward-backward algorithm for the vanilla HMM can be
        found here:
        http://www.cs.columbia.edu/~mcollins/courses/6998-2012/lectures/lec6.3.pdf
        """
        n = len(word)
        forward_values = self.forward(word)
        backward_values = self.backward(word)

        emission_expectations = defaultdict(float)
        transition_expectations = defaultdict(float)

        for prev_tag in ["prefix", "stem", "suffix"]:
            transition_expectations[(prev_tag, "END")] += (
                forward_values[n][prev_tag]
                * self.params.affix_trans_probs[prev_tag]["END"]
            )
        denominator = sum(
            transition_expectations[(prev_tag, "END")]
            for prev_tag in ["prefix", "stem", "suffix"]
        )
        for prev_tag in ["prefix", "stem", "suffix"]:
            transition_expectations[(prev_tag, "END")] /= denominator

        for start in range(n):
            for end in range(start + 1, n + 1):
                substr = word[start:end]
                for prev_tag in ["START", "prefix", "stem", "suffix"]:
                    for cur_tag in ["prefix", "stem", "suffix"]:
                        t = self.params.affix_trans_probs[prev_tag][cur_tag]
                        e = self.params.emission_prob(cur_tag, substr)
                        emission_expectations[(cur_tag, substr)] += (
                            forward_values[start][prev_tag]
                            * t
                            * e
                            * backward_values[end][cur_tag]
                        ) / denominator
                        transition_expectations[(prev_tag, cur_tag)] += (
                            forward_values[start][prev_tag]
                            * t
                            * e
                            * backward_values[end][cur_tag]
                        ) / denominator

        return emission_expectations, transition_expectations

    @staticmethod
    def group_to(max_size, iterable):
        return list(zip_longest(*[iter(iterable)] * max_size, fillvalue=None))

    def expectation_substep(self, words):
        """
        This method is subprocess for the expectation method.
        """
        emissions, transitions, freqs = [], [], []

        for (word, freq) in words:
            e, t = self.forward_backward(word)
            emissions.append(e)
            transitions.append(t)
            freqs.append(freq)

        emission_expectations = defaultdict(float)
        transition_expectations = defaultdict(float)

        for e_key in set(chain(*[list(e.keys()) for e in emissions])):
            emission_expectations[e_key] = sum(
                e[e_key] * freqs[i] for i, e in enumerate(emissions)
            )
        for t_key in set(chain(*[list(t.keys()) for t in transitions])):
            transition_expectations[t_key] = sum(
                t[t_key] * freqs[i] for i, e in enumerate(transitions)
            )
        return emission_expectations, transition_expectations

    def expectation(self, pool, train_words_chunks):
        """
        This method runs the expectation step with a chunked list of training words.
        Args:
            pool: Pool object for multi-threading.
            train_words_chunks: a list of word+frequency-lists (chunked for
                    multi-threading).
        """
        expectations = pool.map(self.expectation_substep, train_words_chunks)

        emission_expectations = {"prefix": {}, "stem": {}, "suffix": {}}
        transition_expectations = {
            "prefix": {},
            "stem": {},
            "suffix": {},
            "START": {},
            "END": {},
        }

        for e_key in set(chain(*[list(e[0].keys()) for e in expectations])):
            emission_expectations[e_key[0]][e_key[1]] = sum(
                e[0][e_key] for e in expectations
            )
        for t_key in set(chain(*[list(t[1].keys()) for t in expectations])):
            transition_expectations[t_key[0]][t_key[1]] = sum(
                t[1][t_key] for t in expectations
            )

        emission_denoms = {
            e: sum(v for v in emission_expectations[e].values())
            for e in emission_expectations.keys()
        }
        transition_denoms = {
            t: sum(v for v in transition_expectations[t].values())
            for t in transition_expectations.keys()
        }
        return (
            emission_expectations,
            emission_denoms,
            transition_expectations,
            transition_denoms,
        )

    def maximization(
        self,
        emission_expectations,
        emission_denoms,
        transition_expectations,
        transition_denoms,
    ):
        """
        Runs the maximization algorithm.
        Args:
            emission_expectations: the expected counts for each affix-morpheme pair.
            emission_denoms: the sum-expected count of each morpheme class.
            transition_expectations: the expected counts for each affix-affix pair
                for transition.
            transition_denoms: the sum-expected count of each morpheme class as
                conditional in transition.
        """
        smoothing_const = self.params.smoothing_const
        for morpheme_class in self.params.morph_emit_probs.keys():
            num_morphs = len(self.params.morph_emit_probs[morpheme_class])
            d = emission_denoms[morpheme_class]
            for morpheme in self.params.morph_emit_probs[morpheme_class].keys():
                e = emission_expectations[morpheme_class][morpheme]
                if d > 0 or smoothing_const > 0:
                    self.params.morph_emit_probs[morpheme_class][morpheme] = (
                        e + smoothing_const
                    ) / ((num_morphs * smoothing_const) + d)
                else:  # for cases of underflowing
                    self.params.morph_emit_probs[morpheme_class][morpheme] = (
                        1.0 / num_morphs
                    )

        for m1 in self.params.affix_trans_probs.keys():
            if m1 == "END":
                continue  # "END" has zero probs for all.
            for m2 in self.params.affix_trans_probs[m1].keys():
                if m2 in transition_expectations[m1]:
                    if transition_denoms[m1] > 0:
                        self.params.affix_trans_probs[m1][m2] = (
                            transition_expectations[m1][m2] / transition_denoms[m1]
                        )
                    else:  # for cases of underflow.
                        self.params.affix_trans_probs[m1][m2] = 1.0 / len(
                            self.params.affix_trans_probs[m1]
                        )
                else:
                    self.params.affix_trans_probs[m1][m2] = 0.0

    def expectation_maximization(self, num_iters, num_cpus=10):
        """
        Runs the EM algorithm.
        Args:
            num_iters: Number of EM epochs.
            num_cpus: Number of cpus for parallel executation of the E step.
        """
        pool = Pool(num_cpus)
        train_words = [
            (word, self.params.word_counts[word])
            for word in self.params.word_counts.keys()
        ]
        chunk_size = math.ceil(float(len(train_words)) / num_cpus)
        train_words_chunks = UnsupervisedMorphology.group_to(chunk_size, train_words)
        for epoch in range(num_iters):
            print("starting epoch %i" % epoch)
            print("starting expectation step")
            ee, ed, te, td = self.expectation(pool, train_words_chunks)
            print("starting maximization step")
            self.maximization(ee, ed, te, td)
            print("updated parameters after maximization")
