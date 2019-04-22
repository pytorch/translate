#!/usr/bin/env python3

import datetime
from collections import Counter, defaultdict
from optparse import OptionParser
from typing import Dict, List, Optional, Set, Tuple


def get_arg_parser():
    parser = OptionParser()
    parser.add_option(
        "--train-file",
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

        # Saving the set of training data indices for each merge candidate. This
        # structure gets updated after every merge operation.
        self.merge_candidate_indices: Dict[Tuple[str, str], Set[int]] = defaultdict(set)

        # Merge candidate is the key, and frequency is the value.
        self.merge_candidate_freq: Dict[Tuple(str, str), float] = defaultdict(float)

    def _init_vocab(self, txt_path: str):
        self.vocab: Dict[str, int] = Counter()
        self.max_bpe_len = 1

        data_freq: Dict[str, int] = Counter()
        with open(txt_path, "r", encoding="utf-8") as input_stream:
            for line in input_stream:
                for word in line.strip().split():
                    # Here, we allow the EOW symbol to be one independent BPE
                    # token. It can potentially attach to previous letters
                    # depending on the frequencies of data. If it is attached,
                    # that is a clear indicator of a suffix.
                    data_freq[" ".join(list(word) + [self.eow_symbol])] += 1

        self.current_train_data: List[Tuple[str, int]] = [None] * len(data_freq)
        for i, (segmentation, freq) in enumerate(data_freq.items()):
            self.current_train_data[i] = (segmentation.split(" "), freq)

        self._init_candidate_frequencies()

    def _init_candidate_frequencies(self) -> None:
        """
        We initialize frequency of candidates. This is the bigrams that are
        extracted from merging pairs of unigrams. This is kept updated after every
        merge operation.
        """
        self.merge_candidate_indices: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        self.merge_candidate_freq: Dict[Tuple(str, str), float] = defaultdict(float)

        for word_index, (seg, freq) in enumerate(self.current_train_data):
            (seg, freq) = self.current_train_data[word_index]
            for i in range(len(seg) - 1):
                self.merge_candidate_freq[(seg[i], seg[i + 1])] += freq
                self.merge_candidate_indices[(seg[i], seg[i + 1])].add(word_index)
                self.vocab[seg[i]] += freq
            self.vocab[seg[-1]] += freq

    def get_best_candidate(self) -> Optional[Tuple[str, str]]:
        return (
            max(self.merge_candidate_freq, key=self.merge_candidate_freq.get)
            if len(self.merge_candidate_freq) > 0
            else None
        )

    def merge_candidate_into_vocab(self, merge_candidate: Tuple[str, str]) -> None:
        """
        Returns Bpe types in the current substep.
        """
        candidate_replacement = "".join(merge_candidate)
        self.max_bpe_len = max(self.max_bpe_len, len(candidate_replacement))
        word_indices_for_merging = set(self.merge_candidate_indices[merge_candidate])

        for word_index in word_indices_for_merging:
            vocab_entry, freq = self.current_train_data[word_index]
            new_entry, current_index = [], 0
            while current_index < len(vocab_entry):
                if (
                    current_index < len(vocab_entry) - 1
                    and (vocab_entry[current_index], vocab_entry[current_index + 1])
                    == merge_candidate
                ):
                    new_entry.append(candidate_replacement)
                    current_index += 2
                else:
                    new_entry.append(vocab_entry[current_index])
                    current_index += 1
            self.current_train_data[word_index] = (new_entry, freq)
            self.update_candidate_frequencies(
                data_index=word_index, old_tokens=vocab_entry, new_tokens=new_entry
            )

    def update_candidate_frequencies(
        self, data_index: int, old_tokens: List[str], new_tokens: List[str]
    ) -> int:
        """
        After each merge operation, we have to update the frequencies of the BPE
        candiates, including the ones that are deprecated (old_tokens), and the
        new ones (new_tokens) with respect to a training word (in data_index).
        """
        freq = self.current_train_data[data_index][1]
        for i in range(len(new_tokens) - 1):
            self.vocab[new_tokens[i]] += freq
            bpe_candidate = (new_tokens[i], new_tokens[i + 1])
            self.merge_candidate_freq[bpe_candidate] += freq
            self.merge_candidate_indices[bpe_candidate].add(data_index)

        self.vocab[new_tokens[-1]] += freq

        for i in range(len(old_tokens) - 1):
            self.vocab[old_tokens[i]] -= freq
            if self.vocab[old_tokens[i]] == 0:
                del self.vocab[old_tokens[i]]

            bpe_candidate = (old_tokens[i], old_tokens[i + 1])

            self.merge_candidate_freq[bpe_candidate] -= freq
            if self.merge_candidate_freq[bpe_candidate] == 0:
                del self.merge_candidate_freq[bpe_candidate]
                del self.merge_candidate_indices[bpe_candidate]

        self.vocab[old_tokens[-1]] -= freq
        if self.vocab[old_tokens[-1]] == 0:
            del self.vocab[old_tokens[-1]]

    def build_vocab(self, txt_path: str, vocab_size: int) -> int:
        """
        After building the vocab, sends the current number of bpe types.

        Args:
            txt_path: Raw text file.
            vocab_size: The maximum number of vocabulary items we need to have.
        """
        self._init_vocab(txt_path=txt_path)
        return self._build_vocab_loop(vocab_size=vocab_size)

    def _build_vocab_loop(self, vocab_size: int) -> int:
        step = 0
        while True:
            merge_candidate = self.get_best_candidate()
            if merge_candidate is not None:
                self.merge_candidate_into_vocab(merge_candidate=merge_candidate)
                if len(self.vocab) >= vocab_size:
                    break
            else:
                # No more merges possible
                break
            step += 1
            if step % 50 == 0:
                print(
                    str(datetime.datetime.now()),
                    "BPE merging step",
                    step,
                    "data size",
                    len(self.current_train_data),
                    "current vocabulary size",
                    len(self.vocab),
                )
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
