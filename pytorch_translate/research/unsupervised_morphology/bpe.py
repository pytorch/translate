#!/usr/bin/env python3

from collections import defaultdict
from typing import Dict


class BPE(object):
    """
    Reimplementation of BPE from https://fburl.com/r69o1rpr (Algorithm 1).
    """

    def __init__(self):
        self.vocab: Dict[str, float] = defaultdict(float)
        self.eow_symbol = "_EOW"  # End of word symbol.
