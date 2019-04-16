#!/usr/bin/env python3

import math
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Tuple


class IBMModel1(object):
    def __init__(self):
        """
        translation_prob is the translation probability in the IBM model 1.
        the full pseudo-code is available at https://fburl.com/yvp31kuw
        """
        self.translation_prob = defaultdict()
        self.null_str = "<null>"
        self.training_data = []

    def initialize_translation_probs(self, src_path: str, dst_path: str):
        """
        Direction of translation is conditioned on the source text: t(dst|src).
        """
        self.training_data = []
        with open(src_path) as src_file, open(dst_path) as dst_file:
            for src_line, dst_line in zip(src_file, dst_file):
                src_words = set(src_line.strip().split() + [self.null_str])
                dst_words = set(dst_line.strip().split())
                for src_word in src_words:
                    if src_word not in self.translation_prob:
                        self.translation_prob[src_word] = defaultdict(float)
                    for dst_word in dst_words:
                        self.translation_prob[src_word][dst_word] = 1.0
                self.training_data.append((src_words, dst_words))

        for src_word in self.translation_prob.keys():
            denom = len(self.translation_prob[src_word])
            for dst_word in self.translation_prob[src_word].keys():
                self.translation_prob[src_word][dst_word] = 1.0 / denom

    def learn_ibm_parameters(
        self, src_path: str, dst_path: str, num_iters: int, num_cpus: int
    ):
        """
        Runs the EM algorithm for IBM model 1.
        Args:
            num_iters: Number of EM iterations.
        """
        self.initialize_translation_probs(src_path=src_path, dst_path=dst_path)
        for iter in range(num_iters):
            print("Iteration of IBM model(1):", str(iter + 1))
            self.em_step(src_path=src_path, dst_path=dst_path, num_cpus=num_cpus)

    def em_step(self, src_path: str, dst_path: str, num_cpus: int) -> None:
        translation_expectations = defaultdict()

        with Pool(processes=num_cpus) as pool:
            data_chunk_size = max(1, math.ceil(len(self.training_data) / num_cpus))
            indices = [
                (
                    i * data_chunk_size,
                    min(data_chunk_size * (i + 1), len(self.training_data)),
                )
                for i in range(num_cpus)
            ]
            results = pool.map(self.e_sub_step, indices)
            for result in results:
                for src_word in result.keys():
                    if src_word not in translation_expectations:
                        translation_expectations[src_word] = defaultdict(float)
                    for dst_word in result[src_word].keys():
                        translation_expectations[src_word][dst_word] += result[
                            src_word
                        ][dst_word]

        self.m_step(translation_expectations)

    def e_sub_step(self, start_end_index: Tuple[int, int]) -> Dict[str, Dict]:
        """
            Running the E step as a substep on the training data based on the
            start and end indices.
        """
        translation_expectations: Dict[str, Dict] = defaultdict()
        start_index, end_index = start_end_index
        for i in range(start_index, end_index):
            src_words, dst_words = self.training_data[i]
            self.e_step(src_words, dst_words, translation_expectations)
        return translation_expectations

    def e_step(
        self, src_words: List, dst_words: List, translation_expectations: Dict
    ) -> None:
        """
        Args:
            translation_expectations: holder of expectations until now. This method
                should update this
        """
        denom = defaultdict(float)
        nominator = defaultdict(float)

        for src_word in src_words:
            if src_word not in nominator:
                nominator[src_word] = defaultdict(float)

            for dst_word in dst_words:
                denom[src_word] += self.translation_prob[src_word][dst_word]
                nominator[src_word][dst_word] += self.translation_prob[src_word][
                    dst_word
                ]

        for src_word in nominator.keys():
            if src_word not in translation_expectations:
                translation_expectations[src_word] = defaultdict(float)
            for dst_word in nominator[src_word].keys():
                delta = nominator[src_word][dst_word] / denom[src_word]
                translation_expectations[src_word][dst_word] += delta

    def m_step(self, translation_expectations) -> None:
        for src_word in translation_expectations.keys():
            denom = sum(translation_expectations[src_word].values())
            for dst_word in translation_expectations[src_word].keys():
                self.translation_prob[src_word][dst_word] = (
                    translation_expectations[src_word][dst_word] / denom
                )
