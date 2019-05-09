#!/usr/bin/env python3

import logging
import math
import pickle
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
        # Integer is the unique integer key for a string from str2int dict.
        self.translation_prob: Dict[int, Dict[int, float]] = defaultdict()
        self.null_str = "<null>"

        # Maps strings to unique integer values.
        self._str2int: Dict[str, int] = {self.null_str: 0}
        self._int2str: List[str] = [self.null_str]  # Reverse of str2int

    def str2int(self, substr: str) -> int:
        if substr in self._str2int:
            return self._str2int[substr]
        else:
            self._str2int[substr] = len(self._str2int)
            self._int2str.append(substr)
            return self._str2int[substr]

    def int2str(self, idx: int) -> str:
        assert idx < len(self._int2str)
        return self._int2str[idx]

    @staticmethod
    def get_word_counts_for_line(self, line: str) -> Dict[int, int]:
        return sum([self.str2int(word) for word in line.strip().split()], Counter())

    def _src_words_counts_in_line(self, line: str) -> Dict[str, int]:
        return Counter(self.str2int(w) for w in line.strip().split() + [self.null_str])

    def _dst_words_counts_in_line(self, line: str) -> Dict[str, int]:
        return Counter(self.str2int(w) for w in line.strip().split())

    def initialize_translation_probs(self, src_path: str, dst_path: str):
        """
        Direction of translation is conditioned on the source text: t(dst|src).
        """
        i = 0
        with open(src_path) as src_file, open(dst_path) as dst_file:
            for src_line, dst_line in zip(src_file, dst_file):
                src_words = self._src_words_counts_in_line(src_line)
                dst_words = self._dst_words_counts_in_line(dst_line)
                for src_word in src_words:
                    if src_word not in self.translation_prob:
                        self.translation_prob[src_word] = defaultdict(float)
                    for dst_word in dst_words:
                        self.translation_prob[src_word][dst_word] = 1.0
                i += 1
                if i % 100000 == 0:
                    logger.info(f"Read sentence {str(i)}")

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
            self.m_step(self.e_step(src_path=src_path, dst_path=dst_path))

    def e_step(self, src_path: str, dst_path: str) -> Dict[str, Dict]:
        logger.info("E step")
        translation_expectations: Dict[str, Dict] = defaultdict()
        i = 0
        with open(src_path) as src_file, open(dst_path) as dst_file:
            for src_line, dst_line in zip(src_file, dst_file):
                src_words = self._src_words_counts_in_line(src_line)
                dst_words = self._dst_words_counts_in_line(dst_line)
                self.expectation_for_one_sentence(
                    src_words, dst_words, translation_expectations
                )
                i += 1
                if i % 1000 == 0:
                    logger.info(f"E step on sentence {str(i)}")
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

    def save(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump((self.translation_prob, self._int2str, self._str2int), f)

    def load(self, file_path: str) -> None:
        with open(file_path, "rb") as f:
            translation_prob, _int2str, _str2int = pickle.load(f)
        self.translation_prob = translation_prob
        self._int2str = _int2str
        self._str2int = _str2int
