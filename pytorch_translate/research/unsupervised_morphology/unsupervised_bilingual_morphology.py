#!/usr/bin/env python3

import copy
import math
import pickle
from collections import defaultdict
from itertools import chain
from multiprocessing import Pool
from typing import Dict, Tuple

from pytorch_translate.research.unsupervised_morphology.unsupervised_morphology import (
    MorphologyHMMParams,
    MorphologySegmentor,
    UnsupervisedMorphology,
)


class BilingualMorphologyHMMParams(MorphologyHMMParams):
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
        self.null_symbol = "_null_"

    def init_params_from_data(self, source_txt_file, target_txt_file):
        """
        We should obtain a list of all possible morphemes from parallel data.
        Args:
            source_txt_file: Source text file with the same number of line as
                target_txt_file.
            target_txt_file: Target text file of the parallel data.
        """
        super().init_params_from_data(input_file_path=source_txt_file)

        target_morphs = {self.null_symbol}  # Should have null translation.
        with open(target_txt_file, "r", encoding="utf-8") as input_stream:
            for line in input_stream:
                sen = line.strip()
                for i, _ in enumerate(sen):
                    for j in range(i, min(i + self.max_morph_len, len(sen))):
                        target_morphs.add(sen[i : j + 1])

        # We just initialize the translation probabilities uniformly.
        self.translation_probs: Dict[str, Dict] = defaultdict()
        uniform_target_probability = {
            morph: 1.0 / len(target_morphs) for morph in target_morphs
        }
        for source_morph in self.morph_emit_probs.keys():
            self.translation_probs[source_morph] = copy.deepcopy(
                uniform_target_probability
            )

    def zero_out_params(self):
        """
        Resets parameter values for all parameters.
        """
        super().zero_out_params()
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

    def get_morpheme_counts(
        self, sentence: str, take_log: bool = False, include_null: bool = True
    ):
        """
        This methods first counts all possible morphemes in a sentence, and then
        returns their fractional count.
        """
        s_len = len(sentence)
        # We first calculate the relative count of each morpheme in the target
        # language.
        target_morpheme_counts: Dict[str, float] = defaultdict(float)
        for start in range(s_len):
            # loop over end indices of morphemes.
            for end in range(start + 1, min(s_len + 1, start + self.max_morph_len)):
                substr = sentence[start:end]
                target_morpheme_counts[substr] += 1
        if include_null:
            target_morpheme_counts[self.null_symbol] = 1

        if take_log is True:
            for morph in target_morpheme_counts.keys():
                target_morpheme_counts[morph] = math.log(target_morpheme_counts[morph])

        return target_morpheme_counts


class BilingualMorphologySegmentor(MorphologySegmentor):
    def segment_blingual_viterbi(self, src_sentence: str, dst_sentence: str):
        """
        This is a dynamic programming algorithm for segmenting a sentence by using
        a modified version of the Viterbi algorithm that also uses the IBM model.
        The main differences to MorphologySegmentor: it uses IBM model parameters
        to find the best segmentation.
        """
        src_len = len(src_sentence)
        pi = [self.params.SMALL_CONST for _ in range(src_len + 1)]
        pi[0] = 0
        back_pointer = [(0, "") for _ in range(src_len + 1)]

        target_morpheme_log_probs = self.params.get_morpheme_counts(
            dst_sentence, take_log=True, include_null=True
        )

        # loop over starting indices of morphemes.
        for src_start in range(src_len):
            # loop over end indices of morphemes.
            for src_end in range(
                src_start + 1, min(src_len + 1, src_start + self.params.max_morph_len)
            ):
                substr = src_sentence[src_start:src_end]
                e = self.params.emission_log_prob(substr)

                # Getting max prability with respect to the target side.
                max_target_prob = self.params.SMALL_CONST
                max_target_morpheme = self.params.null_symbol
                for morpheme in target_morpheme_log_probs.keys():
                    t = target_morpheme_log_probs[
                        morpheme
                    ] + self.params.translation_log_prob(substr, morpheme)
                    if t >= max_target_prob:
                        max_target_prob = t
                        max_target_morpheme = morpheme

                log_prob = pi[src_start] + e + max_target_prob
                if log_prob > pi[src_end]:
                    pi[src_end] = log_prob
                    # Saving backpointer for previous tag and index.
                    back_pointer[src_end] = src_start, max_target_morpheme

        # finalizing the best segmentation.
        indices = [src_len]  # backtracking indices for segmentation.
        target_morphs = [back_pointer[-1][1]]
        indices.append(back_pointer[-1][0])
        while True:
            last_index = indices[-1]
            if last_index == 0:
                break
            start_index = back_pointer[last_index][0]
            indices.append(start_index)
            target_morphs.append(back_pointer[last_index][1])
            if start_index == 0:
                break

        # We should now reverse the backtracked list.
        indices.reverse()
        target_morphs.reverse()
        return indices, target_morphs


class UnsupervisedBilingualMorphology(UnsupervisedMorphology):
    def __init__(
        self,
        src_file: str,
        dst_file: str,
        smoothing_const: float = 0.1,
        use_hardEM: bool = False,
        max_morph_len: int = 8,
        len_cost_pow: float = 2.0,
    ):
        """
        Args:
            use_hardEM: Choosing between soft EM or Viterbi EM (hard EM) algorithm.
        """
        self.params = BilingualMorphologyHMMParams(
            smoothing_const=smoothing_const,
            max_morph_len=max_morph_len,
            len_cost_pow=len_cost_pow,
        )
        self.use_hardEM = use_hardEM
        self.params.init_params_from_data(src_file, dst_file)
        self.segmentor = (
            BilingualMorphologySegmentor(self.params) if self.use_hardEM else None
        )

    def get_translation_marginal(self, src_sentence, dst_sentence):
        """
        This function gets the expected values for translating from morphemes in
        the src_sentence wrt the dst_sentence.
        """
        dst_morph_counts = self.params.get_morpheme_counts(dst_sentence, take_log=False)
        translation_marginal = {}
        n = len(src_sentence)
        for start in range(n):
            for end in range(start + 1, min(n + 1, start + self.params.max_morph_len)):
                src_morph = src_sentence[start:end]
                if src_morph not in translation_marginal:
                    e = self.params.emission_prob(src_morph)
                    translation_sum = sum(
                        dst_morph_counts[dst_morph]
                        * self.params.translation_prob(src_morph, dst_morph)
                        for dst_morph in dst_morph_counts.keys()
                    )
                    translation_marginal[src_morph] = translation_sum * e
        return translation_marginal, dst_morph_counts

    def forward(self, sentence, translation_marginal):
        """
        For the forward pass in the forward-backward algorithm.
        The forward pass is very similar to the Viterbi algorithm. The main difference
        is that here we use summation instead of argmax.
        """
        n = len(sentence)
        forward_values = UnsupervisedMorphology.init_forward_values(n)

        for start in range(n):
            for end in range(start + 1, min(n + 1, start + self.params.max_morph_len)):
                t = translation_marginal[sentence[start:end]]
                forward_values[end] += forward_values[start] * t
        return forward_values

    def backward(self, sentence, translation_marginal):
        """
        For the backward pass in the forward-backward algorithm.
        """
        n = len(sentence)
        backward_values = UnsupervisedMorphology.init_backward_values(n)

        for start in range(n - 1, -1, -1):
            for end in range(start + 1, min(n + 1, start + self.params.max_morph_len)):
                t = translation_marginal[sentence[start:end]]
                backward_values[start] += backward_values[end] * t
        return backward_values

    def forward_backward(self, src_sentence, dst_sentence):
        """
        Similar to the one in UnsupervisedMorphology but it also takes into account
        the target sentence.
        """
        src_len = len(src_sentence)
        translation_marginal, dst_morph_counts = self.get_translation_marginal(
            src_sentence, dst_sentence
        )
        forward_values = self.forward(src_sentence, translation_marginal)
        backward_values = self.backward(src_sentence, translation_marginal)

        emission_expectations: Dict[str, float] = defaultdict(float)
        translation_expectations: Dict[Tuple, Dict] = defaultdict(float)
        fb_expectations: Dict[str, float] = defaultdict(float)
        src_morphemes = set()
        for start in range(src_len):
            for end in range(
                start + 1, min(src_len + 1, start + self.params.max_morph_len)
            ):
                src_morph = src_sentence[start:end]
                src_morphemes.add(src_morph)
                e = self.params.emission_prob(src_morph)
                t_marginal = translation_marginal[src_morph]

                # On morph expectation, we should take care of target translations.
                # That's why we have t_marginal instead of emission_prob.
                emission_expectations[src_morph] += (
                    forward_values[start] * t_marginal * backward_values[end]
                )
                fb_expectations[src_morph] += (
                    forward_values[start] * e * backward_values[end]
                )

        # E(t(dst_morph|src_morph)) = \sum_{start,end}[(t(dst_morph|src_morph))
        # * count(dst_morph) * e(src_morph) * forward(start)*backward(end)]/ Z
        # where Z is a normalization value.
        for src_morph in fb_expectations.keys():
            Z = 0.0
            for dst_morph in dst_morph_counts.keys():
                v = (
                    self.params.translation_prob(src_morph, dst_morph)
                    * dst_morph_counts[dst_morph]
                    * fb_expectations[src_morph]
                )
                translation_expectations[(src_morph, dst_morph)] = v
                Z += v
            for dst_morph in dst_morph_counts.keys():
                translation_expectations[(src_morph, dst_morph)] /= Z

        return emission_expectations, translation_expectations

    def get_expectations_from_viterbi(self, src_sentence, dst_sentence):
        """
        This method segments a sentence with the Viterbi algorithm, and assumes that
        the output of the Viterbi algorithm has an expected count of one, and others
        will have an expected count of zero. This is in contrast with the
        forward-backward algorithm where every possible segmentation is taken into
        account.
        """
        emission_expectations = defaultdict(float)
        translation_expectations: Dict[Tuple, Dict] = defaultdict(float)

        indices, target_morphs = self.segmentor.segment_blingual_viterbi(
            src_sentence, dst_sentence
        )

        for i in range(len(indices) - 1):
            substr = src_sentence[indices[i] : indices[i + 1]]
            emission_expectations[substr] += 1
            translation_expectations[(substr, target_morphs[i])] += 1

        return emission_expectations, translation_expectations

    def expectation_substep(self, sentence_pairs):
        """
        This method is subprocess for the expectation method.
        """
        emission_expectations = defaultdict(float)
        translation_expectations: Dict[Tuple, Dict] = defaultdict(float)

        for sen_pair in sentence_pairs:
            if sen_pair is None:
                continue
            src_sentence, dst_sentence = sen_pair

            if self.use_hardEM:
                e, t = self.get_expectations_from_viterbi(src_sentence, dst_sentence)
            else:
                e, t = self.forward_backward(src_sentence, dst_sentence)

            for e_key in e.keys():
                emission_expectations[e_key] += e[e_key]

            for t_key in t.keys():
                translation_expectations[t_key] += t[t_key]

        return emission_expectations, translation_expectations

    def expectation(self, pool, train_sens_chunks):
        """
        This method runs the expectation step with a chunked list of training
            sentences.
        Args:
            pool: Pool object for multi-threading.
            train_sens_chunks: a list of sentences (chunked for multi-threading).
        """
        expectations = pool.map(self.expectation_substep, train_sens_chunks)
        emission_expectations: Dict[str, float] = defaultdict(float)
        translation_expectations: Dict[Tuple, float] = defaultdict(float)
        for e_key in set(chain(*[list(e[0].keys()) for e in expectations])):
            emission_expectations[e_key] = sum(e[0][e_key] for e in expectations)
        for t_key in set(chain(*[list(e[1].keys()) for e in expectations])):
            translation_expectations[t_key] = sum(e[1][t_key] for e in expectations)

        return emission_expectations, translation_expectations

    def maximization(self, emission_expectations, translation_expectations):
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

        for morpheme in self.params.translation_probs.keys():
            d = sum(translation_expectations[morpheme].values())
            num_dst_morphs = len(self.params.translation_probs[morpheme])
            for dst_morph in self.params.translation_probs[morpheme].keys():
                t = self.params.translation_probs[morpheme][dst_morph]
                if d > 0 or smoothing_const > 0:
                    self.params.translation_probs[morpheme][dst_morph] = (
                        t + smoothing_const
                    ) / ((num_dst_morphs * smoothing_const) + d)
                else:  # for cases of underflowing
                    self.params.translation_probs[morpheme][dst_morph] = (
                        1.0 / num_dst_morphs
                    )

    def expectation_maximization(
        self,
        src_file_path: str,
        dst_file_path: str,
        num_iters: int,
        num_cpus: int = 10,
        model_path: str = None,
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
            self.em_step(pool, src_file_path, dst_file_path, num_cpus, model_path)

    def em_step(self, pool, src_file_path, dst_file_path, num_cpus, model_path):
        """
        One EM step of the EM algorithm.
        """
        print("starting expectation step")
        train_sentences = []
        src_morph_expectations = defaultdict(float)
        translation_expectations: Dict[str, Dict] = defaultdict(dict)
        dst_file_reader = open(dst_file_path, "r")
        with open(src_file_path, "r") as train_open:
            for line in train_open:
                target_sentece = dst_file_reader.readline().strip()
                train_sentences.append((line.strip(), target_sentece))
                # Not allowing to keep many sentences in memory.
                if len(train_sentences) > 100000:
                    chunk_size = math.ceil(float(len(train_sentences)) / num_cpus)
                    train_sens_chunks = UnsupervisedMorphology.group_to(
                        chunk_size, train_sentences
                    )
                    ee, tt = self.expectation(pool, train_sens_chunks)
                    for key in ee.keys():
                        src_morph_expectations[key] += ee[key]
                    for key in tt.keys():
                        src_morph, dst_morph = key
                        if src_morph not in translation_expectations:
                            translation_expectations[src_morph]: Dict[
                                str, float
                            ] = defaultdict(float)
                        translation_expectations[src_morph][dst_morph] += tt[key]
                    train_sentences = []
        if len(train_sentences) > 0:
            chunk_size = math.ceil(float(len(train_sentences)) / num_cpus)
            train_sens_chunks = UnsupervisedMorphology.group_to(
                chunk_size, train_sentences
            )
            ee, tt = self.expectation(pool, train_sens_chunks)
            for key in ee.keys():
                src_morph_expectations[key] += ee[key]
            for key in tt.keys():
                src_morph, dst_morph = key
                if src_morph not in translation_expectations:
                    translation_expectations[src_morph]: Dict[str, float] = defaultdict(
                        float
                    )
                translation_expectations[src_morph][dst_morph] += tt[key]
            train_sentences = []

        dst_file_reader.close()

        print("starting maximization step")
        self.maximization(src_morph_expectations, translation_expectations)
        print("updated parameters after maximization")
        if model_path is not None:
            self.save(model_path)
