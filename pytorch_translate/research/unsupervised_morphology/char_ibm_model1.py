#!/usr/bin/env python3

from collections import Counter
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
