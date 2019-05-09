#!/usr/bin/env python3

import logging
from collections import Counter, defaultdict
from itertools import chain
from optparse import OptionParser
from typing import Dict, List

from pytorch_translate.research.unsupervised_morphology import ibm_model1


logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def get_arg_parser():
    parser = OptionParser()
    parser.add_option(
        "--src-file",
        dest="src_file",
        help="Source file in training data.",
        metavar="FILE",
        default=None,
    )
    parser.add_option(
        "--dst-file",
        dest="dst_file",
        help="Target file in training data.",
        metavar="FILE",
        default=None,
    )
    parser.add_option(
        "--iters",
        type="int",
        dest="ibm_iters",
        help="Number of epochs for training the ibm model.",
        default=3,
    )
    parser.add_option(
        "--num-cpus",
        type="int",
        dest="num_cpus",
        help="Number of cpus for multi-processing.",
        default=3,
    )
    return parser


class CharIBMModel1(ibm_model1.IBMModel1):
    def __init__(self, max_subword_len: int = 8):
        super().__init__()
        self.eow_symbol = "_EW"  # End of word symbol.
        self.max_subword_len = max_subword_len
        self.subword_dict: Dict[int, List[int]] = {}

    def get_possible_subwords(self, word: str) -> List[str]:
        if word in self._str2int and self.str2int(word) in self.subword_dict:
            return self.subword_dict[self.str2int(word)]
        subwords: List[int] = []
        chars = list(word) + [self.eow_symbol]
        for i in range(len(chars)):
            for j in range(i + 1, min(i + 1 + self.max_subword_len, len(chars) + 1)):
                subword = "".join(chars[i:j])
                subwords.append(self.str2int(subword))
        self.subword_dict[self.str2int(word)] = subwords
        return subwords

    def get_subword_counts_for_line(self, line: str) -> Dict[str, int]:
        return Counter(
            chain(*[self.get_possible_subwords(w) for w in line.strip().split()])
        )

    def _src_words_counts_in_line(self, line: str) -> Dict[str, int]:
        return self.get_subword_counts_for_line(line)

    def _dst_words_counts_in_line(self, line: str) -> Dict[str, int]:
        return self.get_subword_counts_for_line(line)


class Word2CharIBMModel1(CharIBMModel1):
    """
    This is similar to the subword-based IBM model but with the exception that
    the source side is still word-based.
    """

    def __init__(self, max_subword_len: int = 8):
        super().__init__(max_subword_len=max_subword_len)

    def _src_words_counts_in_line(self, line: str) -> Dict[str, int]:
        return Counter(self.str2int(w) for w in line.strip().split() + [self.null_str])


if __name__ == "__main__":
    arg_parser = get_arg_parser()
    options, args = arg_parser.parse_args()
    ibm_model = Word2CharIBMModel1()
    ibm_model.learn_ibm_parameters(
        src_path=options.src_file,
        dst_path=options.dst_file,
        num_iters=options.ibm_iters,
    )
