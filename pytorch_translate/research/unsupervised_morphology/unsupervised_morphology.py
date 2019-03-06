#!/usr/bin/env python3

import math
import pickle
import random
from collections import Counter, defaultdict
from itertools import chain, zip_longest
from multiprocessing import Pool
from typing import Dict


class MorphologyHMMParams(object):
    def __init__(
        self,
        smoothing_const: float = 0.1,
        max_morph_len: int = 8,
        len_cost_pow: float = 2,
    ):
        """
        This class contains unigram HMM probabilities for the morphological model.
        Args:
            * smoothing_const: For smoothing the categorical distribution. This is
            mostly useful for unseen observations outside training.
            * max_morph_len: maximum allowed size of a morpheme.
            * len_cost_pow: used for penalizing long char sequences. Here we use
                it in emission as exp(- math.pow(len(str)-1, len_cost_pow))
        """
        self.emit_probs: Dict[str, float] = defaultdict(float)
        self.smoothing_const = smoothing_const
        self.SMALL_CONST = -10000
        self.len_cost_pow = len_cost_pow
        self.max_morph_len = max_morph_len

    def init_params_from_data(self, input_file_path):
        """
        We should obtain a list of all possible morphemes from a data file.
        """
        self.morph_emit_probs: Dict[str, dict] = defaultdict(float)

        with open(input_file_path, "r", encoding="utf-8") as input_stream:
            for line in input_stream:
                sen = line.strip()
                for i in range(0, len(sen)):
                    for j in range(i, min(i + self.max_morph_len, len(sen))):
                        substr = sen[i : j + 1]
                        self.morph_emit_probs[substr] += 1

        # Normalizing the initial probabilities.
        denom = sum(self.morph_emit_probs.values())
        for morph in self.morph_emit_probs.keys():
            self.morph_emit_probs[morph] = (
                self.morph_emit_probs[morph] + self.smoothing_const
            ) / (denom * (1 + self.smoothing_const))

    def zero_out_parmas(self):
        """
        Resets parameter values for all parameters.
        """
        for morpheme in self.morph_emit_probs.keys():
            self.morph_emit_probs[morpheme] = 0.0

    def emission_prob(self, morpheme):
        """
        If a morpheme is not seen, we return the default smoothed probability.
        Note that we assume the probabilities for the seen morphemes are already
        smoothed.
        """
        e = self.morph_emit_probs[morpheme]
        if e == 0:
            return e

        # panalty term for length
        ln = math.pow(len(morpheme) - 1, self.len_cost_pow)
        return e * math.exp(-ln)

    def emission_log_prob(self, morpheme):
        if self.emission_prob(morpheme) == 0:
            return self.SMALL_CONST
        return math.log(self.emission_prob(morpheme))

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            e, s, lc, mml = pickle.load(f)
        m = MorphologyHMMParams(s)
        m.morph_emit_probs = e
        m.len_cost_pow = lc
        m.max_morph_len = mml
        return m

    def save(self, file_path):
        e, s, lc, mml = (
            self.morph_emit_probs,
            self.smoothing_const,
            self.len_cost_pow,
            self.max_morph_len,
        )
        with open(file_path, "wb") as f:
            pickle.dump((e, s, lc, mml), f)


class MorphologySegmentor(object):
    def __init__(self, morphology_hmm_params):
        self.params = morphology_hmm_params

    def segment_viterbi(self, sentence):
        """
        This is a dynamic programming algorithm for segmenting a sentence by using a
        modified version of the Viterbi algorithm. The main difference here is that
        the segment-Viterbi algorithm is based on segment-HMM where a state can
        emit more than one output. In our case, a state is a morpheme and an output
        is a substring of a sentence. Because of searching for a substring
        the segment-Viterbi algorithm has a slower run-time. The segment-Viterbi
        has a complexity O(n^2). In our case, the model is context-insensitive.
        A pseudo-code for the vanilla Viterbi algorithm:
        http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf#page=18
        """
        n = len(sentence)
        pi = [self.params.SMALL_CONST for _ in range(n + 1)]
        pi[0] = 0
        back_pointer = [0 for _ in range(n + 1)]
        for start in range(n):  # loop over starting indices of morphemes.
            # loop over end indices of morphemes.
            for end in range(start + 1, min(n + 1, start + self.params.max_morph_len)):
                e = self.params.emission_log_prob(sentence[start:end])
                log_prob = pi[start] + e
                if log_prob > pi[end]:
                    pi[end] = log_prob
                    # Saving backpointer for previous tag and index.
                    back_pointer[end] = start

        # finalizing the best segmentation.
        indices = [n]  # backtracking indices for segmentation.
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

    def segment_sentence(self, sentence, space_letter: str = "<space>"):
        """
        This method segments sentences based on the Viterbi algorithm. Given an
        input string, the algorithm segments that sentence to one or more
        substrings.
        Args:
            space_letter: letter to put in place of space.
        """
        indices = self.segment_viterbi(sentence)
        outputs = []
        for i in range(len(indices) - 1):
            substr = sentence[indices[i] : indices[i + 1]].replace(" ", space_letter)
            outputs.append(substr)

        return " ".join(outputs)


class UnsupervisedMorphology(object):
    def __init__(
        self,
        input_file: str,
        smoothing_const: float = 0.1,
        use_hardEM: bool = False,
        max_morph_len: int = 8,
        len_cost_pow: float = 2.0,
    ):
        """
        Args:
            use_hardEM: Choosing between soft EM or Viterbi EM (hard EM) algorithm.
        """
        self.params = MorphologyHMMParams(
            smoothing_const=smoothing_const,
            max_morph_len=max_morph_len,
            len_cost_pow=len_cost_pow,
        )
        self.use_hardEM = use_hardEM
        self.params.init_params_from_data(input_file)
        self.segmentor = MorphologySegmentor(self.params) if self.use_hardEM else None

    @staticmethod
    def init_forward_values(n):
        forward_values = [0 for _ in range(n + 1)]
        forward_values[0] = 1.0
        return forward_values

    @staticmethod
    def init_backward_values(n):
        backward_values = [0 for _ in range(n + 1)]
        backward_values[n] = 1.0
        return backward_values

    def forward(self, sentence):
        """
        For the forward pass in the forward-backward algorithm.
        The forward pass is very similar to the Viterbi algorithm. The main difference
        is that here we use summation instead of argmax.
        """
        n = len(sentence)
        forward_values = UnsupervisedMorphology.init_forward_values(n)
        for start in range(n):
            for end in range(start + 1, min(n + 1, start + self.params.max_morph_len)):
                e = self.params.emission_prob(sentence[start:end])
                forward_values[end] += forward_values[start] * e
        return forward_values

    def backward(self, sentence):
        """
        For the backward pass in the forward-backward algorithm.
        """
        n = len(sentence)
        backward_values = UnsupervisedMorphology.init_backward_values(n)

        for start in range(n - 1, -1, -1):
            for end in range(start + 1, min(n + 1, start + self.params.max_morph_len)):
                e = self.params.emission_prob(sentence[start:end])
                backward_values[start] += backward_values[end] * e
        return backward_values

    def forward_backward(self, sentence):
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
        n = len(sentence)
        forward_values = self.forward(sentence)
        backward_values = self.backward(sentence)

        emission_expectations = defaultdict(float)

        for start in range(n):
            for end in range(start + 1, min(n + 1, start + self.params.max_morph_len)):
                substr = sentence[start:end]
                e = self.params.emission_prob(substr)
                emission_expectations[substr] += (
                    forward_values[start] * e * backward_values[end]
                )

        return emission_expectations

    @staticmethod
    def group_to(max_size, iterable):
        return list(zip_longest(*[iter(iterable)] * max_size, fillvalue=None))

    def get_expectations_from_viterbi(self, sentence):
        """
        This method segments a sentence with the Viterbi algorithm, and assumes that
        the output of the Viterbi algorithm has an expected count of one, and others
        will have an expected count of zero. This is in contrast with the
        forward-backward algorithm where every possible segmentation is taken into
        account.
        """
        emission_expectations = defaultdict(float)

        indices = self.segmentor.segment_viterbi(sentence)

        for i in range(len(indices) - 1):
            substr = sentence[indices[i] : indices[i + 1]]
            emission_expectations[substr] += 1

        return emission_expectations

    def expectation_substep(self, sens):
        """
        This method is subprocess for the expectation method.
        """
        emission_expectations = defaultdict(float)
        for sen in sens:
            if sen is None:
                continue

            if self.use_hardEM:
                e = self.get_expectations_from_viterbi(sen)
            else:
                e = self.forward_backward(sen)

            for e_key in e.keys():
                emission_expectations[e_key] += e[e_key]

        return emission_expectations

    def expectation(self, pool, train_sens_chunks):
        """
        This method runs the expectation step with a chunked list of training
            sentences.
        Args:
            pool: Pool object for multi-threading.
            train_sens_chunks: a list of sentences (chunked for multi-threading).
        """
        expectations = pool.map(self.expectation_substep, train_sens_chunks)
        emission_expectations = defaultdict(float)
        for e_key in set(chain(*[list(e.keys()) for e in expectations])):
            emission_expectations[e_key] = sum(e[e_key] for e in expectations)
        return emission_expectations

    def maximization(self, emission_expectations):
        """
        Runs the maximization algorithm.
        Args:
            emission_expectations: the expected counts for each morpheme.
        """
        smoothing_const = self.params.smoothing_const
        num_morphs = len(self.params.morph_emit_probs)
        d = sum(emission_expectations.values())
        for morpheme in self.params.morph_emit_probs.keys():
            e = emission_expectations[morpheme]
            if d > 0 or smoothing_const > 0:
                self.params.morph_emit_probs[morpheme] = (e + smoothing_const) / (
                    (num_morphs * smoothing_const) + d
                )
            else:  # for cases of underflowing
                self.params.morph_emit_probs[morpheme] = 1.0 / num_morphs

    def expectation_maximization(
        self, input_file_path, num_iters, num_cpus=10, model_path=None
    ):
        """
        Runs the EM algorithm.
        Args:
            num_iters: Number of EM epochs.
            num_cpus: Number of cpus for parallel executation of the E step.
        """
        pool = Pool(num_cpus)

        for epoch in range(num_iters):
            print("starting epoch %i" % epoch)
            self.em_step(pool, input_file_path, num_cpus, model_path)

    def em_step(self, pool, input_file_path, num_cpus, model_path):
        """
        One EM step of the EM algorithm.
        """
        print("starting expectation step")
        train_sentences = []
        expectations = defaultdict(float)
        with open(input_file_path, "r") as train_open:
            for line in train_open:
                train_sentences.append(line.strip())
                # Not allowing to keep many sentences in memory.
                if len(train_sentences) > 100000:
                    chunk_size = math.ceil(float(len(train_sentences)) / num_cpus)
                    train_sens_chunks = UnsupervisedMorphology.group_to(
                        chunk_size, train_sentences
                    )
                    ee = self.expectation(pool, train_sens_chunks)
                    for key in ee.keys():
                        expectations[key] += ee[key]
                    train_sentences = []
        if len(train_sentences) > 0:
            chunk_size = math.ceil(float(len(train_sentences)) / num_cpus)
            train_sens_chunks = UnsupervisedMorphology.group_to(
                chunk_size, train_sentences
            )
            ee = self.expectation(pool, train_sens_chunks)
            for key in ee.keys():
                expectations[key] += ee[key]
            train_sentences = []

        print("starting maximization step")
        self.maximization(expectations)
        print("updated parameters after maximization")
        if model_path is not None:
            self.save(model_path)
