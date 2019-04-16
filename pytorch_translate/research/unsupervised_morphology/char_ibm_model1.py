#!/usr/bin/env python3

from collections import Counter, defaultdict
from typing import Dict

from pytorch_translate.research.unsupervised_morphology import ibm_model1


class CharIBMModel1(ibm_model1.IBMModel1):
    def __init__(self, max_subword_len: int = 8):
        super().__init__()
        self.eow_symbol = "_EOW"  # End of word symbol.
        self.max_subword_len = max_subword_len

    def get_possible_subwords(self, word: str) -> Dict[str, int]:
        subwords = Counter()
        chars = list(word) + [self.eow_symbol]
        for i in range(len(chars)):
            for j in range(i + 1, min(i + 1 + self.max_subword_len, len(chars) + 1)):
                subword = "".join(chars[i:j])
                subwords[subword] += 1
        return subwords

    def get_subword_counts_for_line(self, line: str) -> Dict[str, int]:
        return sum(
            [self.get_possible_subwords(word) for word in line.strip().split()],
            Counter(),
        )

    def initialize_translation_probs(self, src_path: str, dst_path: str):
        """
        Direction of translation is conditioned on the source text: t(dst|src).
        """
        with open(src_path) as src_file, open(dst_path) as dst_file:
            for src_line, dst_line in zip(src_file, dst_file):
                src_subwords = self.get_subword_counts_for_line(src_line).keys()
                dst_subwords = self.get_subword_counts_for_line(dst_line).keys()

                for src_subword in src_subwords:
                    if src_subword not in self.translation_prob:
                        self.translation_prob[src_subword] = defaultdict(float)
                    for dst_subword in dst_subwords:
                        self.translation_prob[src_subword][dst_subword] = 1.0

        for src_subword in self.translation_prob.keys():
            denom = len(self.translation_prob[src_subword])
            for dst_subword in self.translation_prob[src_subword].keys():
                self.translation_prob[src_subword][dst_subword] = 1.0 / denom
