#!/usr/bin/env python3

from collections import Counter, defaultdict
from optparse import OptionParser
from typing import Dict

from pytorch_translate.research.unsupervised_morphology import ibm_model1


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
        self.eow_symbol = "_EOW"  # End of word symbol.
        self.max_subword_len = max_subword_len

    def get_possible_subwords(self, word: str) -> Dict[str, int]:
        subwords = []
        chars = list(word) + [self.eow_symbol]
        for i in range(len(chars)):
            for j in range(i + 1, min(i + 1 + self.max_subword_len, len(chars) + 1)):
                subword = "".join(chars[i:j])
                subwords.append(subword)
        return Counter(subwords)

    def get_subword_counts_for_line(self, line: str) -> Dict[str, int]:
        return sum(
            [self.get_possible_subwords(word) for word in line.strip().split()],
            Counter(),
        )

    def initialize_translation_probs(self, src_path: str, dst_path: str):
        """
        Direction of translation is conditioned on the source text: t(dst|src).
        """
        self.training_data = []
        with open(src_path) as src_file, open(dst_path) as dst_file:
            for src_line, dst_line in zip(src_file, dst_file):
                src_subwords_counts = self.get_subword_counts_for_line(src_line)
                dst_subwords_counts = self.get_subword_counts_for_line(dst_line)

                for src_subword in src_subwords_counts:
                    if src_subword not in self.translation_prob:
                        self.translation_prob[src_subword] = defaultdict(float)
                    for dst_subword in dst_subwords_counts:
                        self.translation_prob[src_subword][dst_subword] = 1.0
                self.training_data.append((src_subwords_counts, dst_subwords_counts))

        for src_subword in self.translation_prob.keys():
            denom = len(self.translation_prob[src_subword])
            for dst_subword in self.translation_prob[src_subword].keys():
                self.translation_prob[src_subword][dst_subword] = 1.0 / denom


class Word2CharIBMModel1(CharIBMModel1):
    """
    This is similar to the subword-based IBM model but with the exception that
    the source side is still word-based.
    """

    def __init__(self, max_subword_len: int = 8):
        super().__init__(max_subword_len=max_subword_len)

    def initialize_translation_probs(self, src_path: str, dst_path: str):
        """
        Direction of translation is conditioned on the source text: t(dst|src).
        """
        self.training_data = []
        with open(src_path) as src_file, open(dst_path) as dst_file:
            for src_line, dst_line in zip(src_file, dst_file):
                src_words_counts = Counter(src_line.strip().split())
                dst_sub_words = self.get_subword_counts_for_line(dst_line)

                for src_word in src_words_counts.keys():
                    if src_word not in self.translation_prob:
                        self.translation_prob[src_word] = defaultdict(float)
                    for dst_sub_word in dst_sub_words.keys():
                        self.translation_prob[src_word][dst_sub_word] = 1.0
                self.training_data.append((src_words_counts, dst_sub_words))

        for src_word in self.translation_prob.keys():
            denom = len(self.translation_prob[src_word])
            for dst_subword in self.translation_prob[src_word].keys():
                self.translation_prob[src_word][dst_subword] = 1.0 / denom


if __name__ == "__main__":
    arg_parser = get_arg_parser()
    options, args = arg_parser.parse_args()
    ibm_model = CharIBMModel1()
    ibm_model.learn_ibm_parameters(
        src_path=options.src_file,
        dst_path=options.dst_file,
        num_iters=options.ibm_iters,
        num_cpus=options.num_cpus,
    )
