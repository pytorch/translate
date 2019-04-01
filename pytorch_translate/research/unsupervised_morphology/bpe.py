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
