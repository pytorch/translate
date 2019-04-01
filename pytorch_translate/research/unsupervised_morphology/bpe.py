#!/usr/bin/env python3

from collections import Counter
from typing import Dict


class BPE(object):
    """
    Reimplementation of BPE from https://fburl.com/r69o1rpr (Algorithm 1).
    """

    def __init__(self):
        self.vocab: Dict[str, float] = Counter()
        self.eow_symbol = "_EOW"  # End of word symbol.

    def init_vocab(self, txt_path: str):
        self.vocab: Dict[str, float] = Counter()

        with open(txt_path, "r", encoding="utf-8") as input_stream:
            for line in input_stream:
                for word in line.strip().split():
                    self.vocab[" ".join(list(word) + [self.eow_symbol])] += 1

    def get_best_candidate(self):
        """
        Calculates frequencies for new candidiates from the current vocabulary,
        and returns the candidate with the most frequency.
        """
        candidates = Counter()
        for vocab_entry, freq in self.vocab.items():
            symbols = vocab_entry.split()
            for i in range(len(symbols) - 1):
                candidates[(symbols[i], symbols[i + 1])] += freq
        return max(candidates, key=candidates.get)
