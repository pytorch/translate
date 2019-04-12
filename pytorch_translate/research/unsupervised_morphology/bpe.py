#!/usr/bin/env python3

import re
from collections import Counter
from optparse import OptionParser
from typing import Dict, List, Tuple


def get_arg_parser():
    parser = OptionParser()
    parser.add_option(
        "--train",
        dest="train_file",
        help="Raw text as training data.",
        metavar="FILE",
        default=None,
    )
    parser.add_option(
        "--vocab-size",
        type="int",
        dest="vocab_size",
        help="Vocabulary Size.",
        default=20000,
    )
    parser.add_option(
        "--train-out",
        dest="train_output_file",
        help="BPE tokenized train file.",
        metavar="FILE",
        default=None,
    )
    return parser


class BPE(object):
    """
    Reimplementation of BPE from https://fburl.com/r69o1rpr (Algorithm 1).
    """

    def __init__(self):
        self.vocab: Dict[str, int] = Counter()
        self.eow_symbol = "_EOW"  # End of word symbol.

        # This data structure holds current segmentation of training data. This
        # is useful for faster parallel computation during training.
        # str is the current segmentation, int is the frequency.
        self.current_train_data: List[Tuple[str, int]] = []

        # This value will change after building the vocabulary. We use this value
        # for greedy segmentation where we start by looking at the longest possible
        # character sequence wrt max_bpe_len.
        self.max_bpe_len = 1

    def _init_vocab(self, txt_path: str):
        data_freq: Dict[str, int] = Counter()
        with open(txt_path, "r", encoding="utf-8") as input_stream:
            for line in input_stream:
                for word in line.strip().split():
                    # Here, we allow the EOW symbol to be one independent BPE
                    # token. It can potentially attach to previous letters
                    # depending on the frequencies of data. If it is attached,
                    # that is a clear indicator of a suffix.
                    data_freq[" ".join(list(word) + [self.eow_symbol])] += 1

        self.current_train_data: List[Tuple[str, int]] = []
        for segmentation, freq in data_freq.items():
            self.current_train_data.append((segmentation, freq))

    def get_best_candidate(self):
        """
        Calculates frequencies for new candidiates from the current vocabulary,
        and returns the candidate with the most frequency.
        """
        candidates = Counter()
        for (segmentation, freq) in self.current_train_data:
            symbols = segmentation.split()
            for i in range(len(symbols) - 1):
                candidates[(symbols[i], symbols[i + 1])] += freq
        return max(candidates, key=candidates.get) if len(candidates) > 0 else None

    @staticmethod
    def get_merge_pattern(candidate_str):
        return re.compile(r"(?<!\S)" + re.escape(candidate_str) + r"(?!\S)")

    def merge_candidate_into_vocab(self, candidate: Tuple[str, str]) -> int:
        """
        Returns the vocabulary size (number of BPE types).
        Args:
            candidate: a pair of strings to be merged in all entries.
        """
        candidate_str = " ".join(candidate)
        candidate_replacement = "".join(candidate)
        pattern = BPE.get_merge_pattern(candidate_str)

        new_bpe_entries = set()
        for i in range(len(self.current_train_data)):
            vocab_entry, freq = self.current_train_data[i]
            new_entry = vocab_entry
            if candidate_str in vocab_entry:
                # Regex is usually slow. We just apply it on words that have the
                # potential of replacement.
                new_entry = pattern.sub(candidate_replacement, vocab_entry)

            self.current_train_data[i] = (new_entry, freq)
            for entry in new_entry.split():
                new_bpe_entries.add(entry)

        return len(new_bpe_entries)

    def build_vocab(self, txt_path: str, vocab_size: int) -> int:
        """
        After building the vocab, sends the current number of bpe types.

        Args:
            txt_path: Raw text file.
            vocab_size: The maximum number of vocabulary items we need to have.
        """
        self._init_vocab(txt_path=txt_path)
        step = 0
        while True:
            merge_candidate = self.get_best_candidate()
            if merge_candidate is not None:
                cur_v_size = self.merge_candidate_into_vocab(merge_candidate)
                if cur_v_size >= vocab_size:
                    break
            else:
                # No more merges possible
                break
            step += 1
            if step % 100 == 0:
                print("BPE merging step", step, "current vocabulary size", cur_v_size)

        # Now we get rid of the current vocab that is based on the corpus (not
        # memory-efficient). We now only keep the final bpe tokens.
        self.vocab: Dict[str, int] = Counter()
        self.max_bpe_len = 1
        for (vocab_entry, freq) in self.current_train_data:
            for bpe_token in vocab_entry.split():
                self.vocab[bpe_token] += freq
                self.max_bpe_len = max(self.max_bpe_len, len(bpe_token))

        print("BPE vocab built with size", len(self.vocab))
        return len(self.vocab)

    def segment_word(self, word: str) -> List[str]:
        """
        The current segmentation is greedy based on picking the longest possible
        character sequences first. The original work picks based on the most
        frequent character sequence.
        """
        word_chars = list(word) + [self.eow_symbol]
        start_idx, end_idx = 0, min(len(word_chars), self.max_bpe_len)
        subwords = []
        while start_idx < len(word_chars):
            subword = "".join(word_chars[start_idx:end_idx])
            if subword in self.vocab or end_idx - start_idx == 1:
                subwords.append(subword)
                start_idx = end_idx
                end_idx = min(len(word_chars), start_idx + self.max_bpe_len)
            else:
                end_idx -= 1
        return subwords

    def segment_txt(self, input_path: str, output_path: str):
        segmentation_cache = {}
        with open(output_path, "w", encoding="utf-8") as writer:
            with open(input_path, "r", encoding="utf-8") as input_file:
                for line in input_file:
                    output_bpe_tokens = []
                    for word in line.strip().split():
                        if word not in segmentation_cache:
                            segmentation_cache[word] = self.segment_word(word)
                        output_bpe_tokens += segmentation_cache[word]
                    writer.write(" ".join(output_bpe_tokens))
                    writer.write("\n")


if __name__ == "__main__":
    arg_parser = get_arg_parser()
    options, args = arg_parser.parse_args()
    bpe_model = BPE()
    bpe_model.build_vocab(txt_path=options.train_file, vocab_size=options.vocab_size)
    bpe_model.segment_txt(
        input_path=options.train_file, output_path=options.train_output_file
    )
