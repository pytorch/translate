#!/usr/bin/env python3

from fairseq import dictionary
from fbtranslate import vocab_constants


class Dictionary(dictionary.Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        max_special_tokens=vocab_constants.MAX_SPECIAL_TOKENS,
    ):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}

        self.pad_index = self.add_symbol(pad)
        assert self.pad_index == vocab_constants.PAD_ID

        # Adds a junk symbol for vocab_constants' GO_ID
        self.add_symbol('<reserved>')

        self.eos_index = self.add_symbol(eos)
        assert self.eos_index == vocab_constants.EOS_ID

        self.unk_index = self.add_symbol(unk)
        assert self.unk_index == vocab_constants.UNK_ID

        # Adds junk symbols to pad up to the number of special tokens.
        num_reserved = max_special_tokens - len(self.symbols)
        for i in range(num_reserved):
            self.add_symbol(f'<reserved_{i}>')

        self.nspecial = len(self.symbols)
        assert self.nspecial == max_special_tokens


class CharDictionary(Dictionary):
    """Character vocab with its additonal special tokens."""

    def __init__(
        self,
        word_delim='<space>',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.word_delim = word_delim
        self.bow_index = self.add_symbol('<bow>')
        self.eow_index = self.add_symbol('<eow>')
        self.word_delim_index = self.add_symbol(word_delim)
        self.nspecial += 3
