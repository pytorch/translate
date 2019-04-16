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
        self.training_data = []
        with open(src_path) as src_file, open(dst_path) as dst_file:
            for src_line, dst_line in zip(src_file, dst_file):
                src_subwords_counts = self.get_subword_counts_for_line(src_line)
                dst_subwords_counts = self.get_subword_counts_for_line(dst_line)

                for src_subword in src_subwords_counts.keys():
                    if src_subword not in self.translation_prob:
                        self.translation_prob[src_subword] = defaultdict(float)
                    for dst_subword in dst_subwords_counts.keys():
                        self.translation_prob[src_subword][dst_subword] = 1.0
                self.training_data.append((src_subwords_counts, dst_subwords_counts))

        for src_subword in self.translation_prob.keys():
            denom = len(self.translation_prob[src_subword])
            for dst_subword in self.translation_prob[src_subword].keys():
                self.translation_prob[src_subword][dst_subword] = 1.0 / denom

    def e_step(
        self,
        src_words: Dict[str, int],
        dst_words: Dict[str, int],
        translation_expectations: Dict,
    ) -> None:
        """
        Args:
            translation_expectations: holder of expectations until now. This method
                should update this
        """
        denom = defaultdict(float)
        nominator = defaultdict(float)

        for src_word in src_words.keys():
            if src_word not in nominator:
                nominator[src_word] = defaultdict(float)
            src_word_count = src_words[src_word]
            for dst_word in dst_words.keys():
                dst_word_count = dst_words[dst_word]
                num_combinations = src_word_count * dst_word_count
                denom[src_word] += (
                    num_combinations * self.translation_prob[src_word][dst_word]
                )
                nominator[src_word][dst_word] += (
                    num_combinations * self.translation_prob[src_word][dst_word]
                )

        for src_word in nominator.keys():
            if src_word not in translation_expectations:
                translation_expectations[src_word] = defaultdict(float)
            for dst_word in nominator[src_word].keys():
                delta = nominator[src_word][dst_word] / denom[src_word]
                translation_expectations[src_word][dst_word] += delta
