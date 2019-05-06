#!/usr/bin/env python3

import logging
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class IBMModel1(object):
    def __init__(self):
        """
        translation_prob is the translation probability in the IBM model 1.
        the full pseudo-code is available at https://fburl.com/yvp31kuw
        """
        self.translation_prob = defaultdict()
        self.null_str = "<null>"
        self.training_data = []

    @staticmethod
    def get_word_counts_for_line(self, line: str) -> Dict[str, int]:
        return sum(
            [self.get_possible_subwords(word) for word in line.strip().split()],
            Counter(),
        )

    def initialize_translation_probs(self, src_path: str, dst_path: str):
        """
        Direction of translation is conditioned on the source text: t(dst|src).
        """
        self.training_data = []
        with open(src_path) as src_file, open(dst_path) as dst_file:
            for src_line, dst_line in zip(src_file, dst_file):
                src_words = Counter(src_line.strip().split() + [self.null_str])
                dst_words = Counter(dst_line.strip().split())
                for src_word in src_words:
                    if src_word not in self.translation_prob:
                        self.translation_prob[src_word] = defaultdict(float)
                    for dst_word in dst_words:
                        self.translation_prob[src_word][dst_word] = 1.0
                self.training_data.append((src_words, dst_words))
                if len(self.training_data) % 100000 == 0:
                    logger.info(f"Read sentence# {len(self.training_data)}")

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
        logger.info("Initializing model parameters")
        self.initialize_translation_probs(src_path=src_path, dst_path=dst_path)
        for iter in range(num_iters):
            logger.info(f"Iteration of IBM model: {str(iter+1)}")
            self.m_step(self.e_step())

    def e_step(self) -> Dict[str, Dict]:
        logger.info("E step")
        translation_expectations: Dict[str, Dict] = defaultdict()
        for i, (src_words, dst_words) in enumerate(self.training_data):
            self.expectation_for_one_sentence(
                src_words, dst_words, translation_expectations
            )
            if (i + 1) % 1000 == 0:
                logger.info(f"E step on sentence {str(i+1)}")
        return translation_expectations

    def expectation_for_one_sentence(
        self,
        src_words: Dict[str, int],
        dst_words: Dict[str, int],
        translation_expectations: Dict,
    ) -> None:
        """
        Args:
            translation_expectations: holder of expectations until now.
            src_words and dst_words are Counter objects.
        """
        denom = defaultdict(float)
        translation_fractional_counts = defaultdict(lambda: defaultdict(float))
        for src_word in src_words:
            for dst_word in dst_words:
                s_count, d_count = src_words[src_word], dst_words[dst_word]
                prob = self.translation_prob[src_word][dst_word] * s_count * d_count
                denom[dst_word] += prob
                translation_fractional_counts[src_word][dst_word] += prob

        for src_word in translation_fractional_counts.keys():
            if src_word not in translation_expectations:
                translation_expectations[src_word] = defaultdict(float)
            for dst_word in translation_fractional_counts[src_word].keys():
                delta = (
                    translation_fractional_counts[src_word][dst_word] / denom[dst_word]
                )
                translation_expectations[src_word][dst_word] += delta

    def m_step(self, translation_expectations) -> None:
        logger.info("M step")
        for src_word in translation_expectations.keys():
            denom = sum(translation_expectations[src_word].values())
            for dst_word in translation_expectations[src_word].keys():
                self.translation_prob[src_word][dst_word] = (
                    translation_expectations[src_word][dst_word] / denom
                )
