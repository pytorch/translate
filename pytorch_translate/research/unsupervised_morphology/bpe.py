#!/usr/bin/env python3

import logging
import math
from collections import Counter, defaultdict
from multiprocessing import Pool
from optparse import OptionParser
from typing import Dict, List, Optional, Set, Tuple


logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


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
    parser.add_option(
        "--num-cpus",
        type="int",
        dest="num_cpus",
        help="Number of cpus for multi-processing.",
        default=3,
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

        self.current_train_data: List[Tuple[str, int]] = [None] * len(data_freq)
        for i, (segmentation, freq) in enumerate(data_freq.items()):
            self.current_train_data[i] = (segmentation.split(" "), freq)

    def _best_candidate_substep(
        self, start_end_indices: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        Args:
            first and end index for part of self.current_train_data to search for.
        """
        start_index, end_index = start_end_indices[0], start_end_indices[1]
        assert start_index <= end_index

        candidates = defaultdict(float)
        for i in range(start_index, end_index):
            if i >= len(self.current_train_data):
                break
            (seg, freq) = self.current_train_data[i]
            for i in range(len(seg) - 1):
                candidates[(seg[i], seg[i + 1])] += freq
        return candidates

    def get_best_candidate(
        self, num_cpus: int, pool: Pool
    ) -> Optional[Tuple[str, str]]:
        """
        Calculates frequencies for new candidiates from the current vocabulary,
        and returns the candidate with the most frequency.
        """
        data_chunk_size = max(1, math.ceil(len(self.current_train_data) / num_cpus))
        indices = [
            (
                i * data_chunk_size,
                min(data_chunk_size * (i + 1), len(self.current_train_data)),
            )
            for i in range(num_cpus)
        ]
        results = pool.map(self._best_candidate_substep, indices)
        candidates = defaultdict(float)
        for result in results:
            for (k, v) in result.items():
                candidates[k] += v
        return max(candidates, key=candidates.get) if len(candidates) > 0 else None

    def merge_substep(
        self, merge_candidate: Tuple[str, str], start_end_index: Tuple[int, int]
    ) -> Tuple[List[Tuple[str, int]], Set]:
        """
        Returns Bpe types in the current substep.
        """
        candidate_replacement = "".join(merge_candidate)
        offset, stop_index = start_end_index
        assert offset < stop_index

        new_bpe_entries = set()
        new_data: Dict[int, List[Tuple[str, int]]] = {}
        for i in range(offset, stop_index):
            if i >= len(self.current_train_data):
                break
            vocab_entry, freq = self.current_train_data[i]
            new_entry, current_index = [], 0
            while current_index < len(vocab_entry):
                if (
                    current_index < len(vocab_entry) - 1
                    and (vocab_entry[current_index], vocab_entry[current_index + 1])
                    == merge_candidate
                ):
                    new_entry.append(candidate_replacement)
                    new_bpe_entries.add(candidate_replacement)
                    current_index += 2
                else:
                    new_entry.append(vocab_entry[current_index])
                    new_bpe_entries.add(vocab_entry[current_index])
                    current_index += 1

            new_data[i] = (new_entry, freq)

        return (new_data, new_bpe_entries)

    def merge_candidate_into_vocab(
        self, candidate: Tuple[str, str], num_cpus: int, pool: Pool
    ) -> int:
        """
        Returns the vocabulary size (number of BPE types).
        Args:
            candidate: a pair of strings to be merged in all entries.
        """
        data_chunk_size = max(1, math.ceil(len(self.current_train_data) / num_cpus))
        candidate_str_list = [
            (
                (candidate[0], candidate[1]),
                (
                    i * data_chunk_size,
                    min(data_chunk_size * (i + 1), len(self.current_train_data)),
                ),
            )
            for i in range(num_cpus)
        ]

        results = pool.starmap(self.merge_substep, candidate_str_list)

        for result in results:
            for k, v in result[0].items():
                self.current_train_data[k] = v
        bpe_types_union = set.union(*[result[1] for result in results])
        return len(bpe_types_union)

    def build_vocab(self, txt_path: str, vocab_size: int, num_cpus: int) -> int:
        """
        After building the vocab, sends the current number of bpe types.

        Args:
            txt_path: Raw text file.
            vocab_size: The maximum number of vocabulary items we need to have.
        """
        self._init_vocab(txt_path=txt_path)
        return self._build_vocab_loop(vocab_size=vocab_size, num_cpus=num_cpus)

    def _build_vocab_loop(self, vocab_size: int, num_cpus: int) -> int:
        step = 0
        with Pool(processes=num_cpus) as pool:
            while True:
                merge_candidate = self.get_best_candidate(num_cpus=num_cpus, pool=pool)
                if merge_candidate is not None:
                    cur_v_size = self.merge_candidate_into_vocab(
                        candidate=merge_candidate, num_cpus=num_cpus, pool=pool
                    )
                    if cur_v_size >= vocab_size:
                        break
                else:
                    # No more merges possible
                    break
                step += 1
                if step % 50 == 0:
                    logger.warning(
                        f"""BPE merge step: {step}, data size:
                        {len(self.current_train_data)}, vocab size, {cur_v_size}"""
                    )
        return self.finalize_vocab()

    def finalize_vocab(self) -> int:
        # Now we get rid of the current vocab that is based on the corpus (not
        # memory-efficient). We now only keep the final bpe tokens.
        self.vocab: Dict[str, int] = Counter()
        self.max_bpe_len = 1
        for (vocab_entry, freq) in self.current_train_data:
            for bpe_token in vocab_entry:
                self.vocab[bpe_token] += freq
                self.max_bpe_len = max(self.max_bpe_len, len(bpe_token))

        logger.warning(f"BPE vocab built with size {len(self.vocab)}")
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
    bpe_model.build_vocab(
        txt_path=options.train_file,
        vocab_size=options.vocab_size,
        num_cpus=options.num_cpus,
    )
    bpe_model.segment_txt(
        input_path=options.train_file, output_path=options.train_output_file
    )
