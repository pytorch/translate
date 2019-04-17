#!/usr/bin/env python3

from collections import defaultdict
from typing import Dict, List


class IBMModel1(object):
    def __init__(self):
        """
        translation_prob is the translation probability in the IBM model 1.
        the full pseudo-code is available at https://fburl.com/yvp31kuw
        """
        self.translation_prob = defaultdict(lambda: defaultdict(float))
        self.null_str = "<null>"

    def initialize_translation_probs(self, src_path: str, dst_path: str):
        """
        Direction of translation is conditioned on the source text: t(dst|src).
        """
        with open(src_path) as src_file, open(dst_path) as dst_file:
            for src_line, dst_line in zip(src_file, dst_file):
                src_words = set(src_line.strip().split() + [self.null_str])
                dst_words = set(dst_line.strip().split())
                for src_word in src_words:
                    if src_word not in self.translation_prob:
                        self.translation_prob[src_word] = defaultdict(float)
                    for dst_word in dst_words:
                        self.translation_prob[src_word][dst_word] = 1.0

        for src_word in self.translation_prob.keys():
            denom = len(self.translation_prob[src_word])
            for dst_word in self.translation_prob[src_word].keys():
                self.translation_prob[src_word][dst_word] = 1.0 / denom

    def learn_ibm_parameters(self, src_path: str, dst_path: str, num_iters: int):
        """
        Runs the EM algorithm for IBM model 1.
        Args:
            num_iters: Number of EM iterations.
        """
        self.initialize_translation_probs(src_path=src_path, dst_path=dst_path)
        for iter in range(num_iters):
            print("Iteration of IBM model(1):", str(iter + 1))
            self.em_step(src_path=src_path, dst_path=dst_path)

    def em_step(self, src_path: str, dst_path: str):
        translation_expectations = defaultdict()

        with open(src_path) as src_file, open(dst_path) as dst_file:
            for src_line, dst_line in zip(src_file, dst_file):
                src_words = src_line.strip().split() + [self.null_str]
                dst_words = dst_line.strip().split()
                self.e_step(src_words, dst_words, translation_expectations)
        self.m_step(translation_expectations)

    def e_step(self, src_words: List, dst_words: List, translation_expectations: Dict):
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

    def m_step(self, translation_expectations):
        for src_word in translation_expectations.keys():
            denom = sum(translation_expectations[src_word].values())
            for dst_word in translation_expectations[src_word].keys():
                self.translation_prob[src_word][dst_word] = (
                    translation_expectations[src_word][dst_word] / denom
                )
