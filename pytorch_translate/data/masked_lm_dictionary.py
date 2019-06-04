#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


from typing import Dict, List, Set

from pytorch_translate import vocab_constants
from pytorch_translate.data.dictionary import Dictionary


class MaskedLMDictionary(Dictionary):
    """
    Dictionary for Masked Language Modelling tasks. This extends Dictionary by
    adding the mask symbol.
    """

    def __init__(
        self,
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        mask="<mask>",
        max_special_tokens: int = vocab_constants.MAX_SPECIAL_TOKENS,
    ):
        self.symbols: List[str] = []
        self.count: List[int] = []
        self.indices: Dict[str, int] = {}
        self.lexicon_indices: Set[int] = set()

        self.pad_word = pad
        self.pad_index = self.add_symbol(pad)
        assert self.pad_index == vocab_constants.PAD_ID

        # Adds a junk symbol for vocab_constants' GO_ID
        self.add_symbol("<reserved>")

        self.eos_word = eos
        self.eos_index = self.add_symbol(eos)
        assert self.eos_index == vocab_constants.EOS_ID

        self.unk_word = unk
        self.unk_index = self.add_symbol(unk)
        assert self.unk_index == vocab_constants.UNK_ID

        self.mask_word = mask
        self.mask_index = self.add_symbol(mask)
        assert self.mask_index == vocab_constants.MASK_ID

        # Adds junk symbols to pad up to the number of special tokens.
        num_reserved = max_special_tokens - len(self.symbols)
        for i in range(num_reserved):
            self.add_symbol(f"<reserved_{i}>")

        self.nspecial = len(self.symbols)
        assert self.nspecial == max_special_tokens

    def mask(self):
        """Helper to get index of mask symbol"""
        return self.mask_index
