#!/usr/bin/env python3

import logging
from collections import defaultdict
from optparse import OptionParser
from typing import Dict, List, Set, Tuple

from pytorch_translate.research.unsupervised_morphology.bpe import BPE
from pytorch_translate.research.unsupervised_morphology.char_ibm_model1 import (
    Word2CharIBMModel1,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def get_arg_parser():
    parser = OptionParser()
    parser.add_option(
        "--src-file",
        dest="src_train_file",
        help="Source raw text as training data.",
        metavar="FILE",
        default=None,
    )
    parser.add_option(
        "--dst-file",
        dest="dst_train_file",
        help="Target raw text as training data.",
        metavar="FILE",
        default=None,
    )
    parser.add_option(
        "--vocab-size",
        type="int",
        dest="vocab_size",
        help="Source vocabulary Size.",
        default=20000,
    )
    parser.add_option(
        "--train-out",
        dest="train_output_file",
        help="BPE tokenized source train file.",
        metavar="FILE",
        default=None,
    )
    parser.add_option(
        "--ibm-iters",
        type="int",
        dest="num_ibm_iters",
        help="Number of training epochs for character IBM models.",
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


class BilingualBPE(BPE):
    """
    An extension of the BPE model that is cross-lingual wrt parallel data.
    """

    def __init__(self):
        super().__init__()
        self.dst2src_ibm_model = Word2CharIBMModel1()

    def _init_params(
        self, src_txt_path: str, dst_txt_path: str, num_ibm_iters: int, num_cpus: int
    ):
        """
        Args:
            src_txt_path: Text path for source language in parallel data.
            dst_txt_path: Text path for target language in parallel data.
            num_ibm_iters: Number of training epochs for the IBM model.
            num_cpus: Number of CPUs for training the IBM model with multi-processing.
        """
        logger.info("Initializing vocabulary.")

        # Note the reverse side of the model. Target is word based, that is why
        # we give it a reverse order.
        self.dst2src_ibm_model.learn_ibm_parameters(
            src_path=dst_txt_path,
            dst_path=src_txt_path,
            num_iters=num_ibm_iters,
            num_cpus=num_cpus,
        )
        logger.info("calculating alignment-based BPE type probs.")
        self.bpe_probs_from_alignment = self._calc_bpe_prob_from_alignment(
            dst_txt_path=dst_txt_path
        )

        self._init_vocab(txt_path=src_txt_path)

    def _calc_word_probs(self, txt_path: str) -> Dict[str, float]:
        """
        Calculates the probability of each word from raw counts in a text file.
        """
        vocab = defaultdict(float)
        with open(txt_path) as txt_file:
            for line in txt_file:
                for word in line.strip().split():
                    vocab[word] += 1

        denom = sum(vocab.values())
        for word in vocab.keys():
            vocab[word] /= denom
        return vocab

    def _calc_bpe_prob_from_alignment(self, dst_txt_path: str) -> Dict[str, float]:
        """
         p(subword=s) = sum_{t in target} p(s|t) p(t)
         where p(t) is target_word_prob[t] from _calc_word_probs
         and p(s|t) = self.dst2src_ibm_model.translation_prob[t][s]
        """
        target_word_probs = self._calc_word_probs(txt_path=dst_txt_path)
        bpe_alignment_prob = defaultdict(float)
        for dst_word in self.dst2src_ibm_model.translation_prob.keys():
            target_word_prob = target_word_probs[dst_word]
            alignment_probs = self.dst2src_ibm_model.translation_prob[dst_word]
            for src_subword in self.dst2src_ibm_model.translation_prob[dst_word]:
                bpe_alignment_prob[src_subword] += (
                    alignment_probs[src_subword] * target_word_prob
                )
        for src_subword in bpe_alignment_prob.keys():
            bpe_alignment_prob[src_subword] = max(
                bpe_alignment_prob[src_subword], 1e-30
            )
        return bpe_alignment_prob

    def _init_candidate_frequencies(self) -> None:
        self.merge_candidate_indices: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        self.merge_candidate_freq: Dict[Tuple(str, str), float] = defaultdict(float)

        for word_index, (seg, freq) in enumerate(self.current_train_data):
            (seg, freq) = self.current_train_data[word_index]
            for i in range(len(seg) - 1):
                bpe_key = (seg[i], seg[i + 1])
                bpe_token = "".join(bpe_key)
                self.merge_candidate_freq[bpe_key] += self.bpe_alignment_prob(
                    bpe_token, freq
                )
                self.merge_candidate_indices[bpe_key].add(word_index)
                self.vocab[seg[i]] += freq
            self.vocab[seg[-1]] += freq

    def bpe_alignment_prob(self, bpe_token: str, freq: int):
        if bpe_token in self.bpe_probs_from_alignment:
            return self.bpe_probs_from_alignment[bpe_token]
        else:
            # In cases where the alignment model did not cover long character
            # sequences in training data.
            return freq * 1e-30

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
            bpe_token = "".join(bpe_candidate)
            self.merge_candidate_freq[bpe_candidate] += self.bpe_alignment_prob(
                bpe_token, freq
            )
            self.merge_candidate_indices[bpe_candidate].add(data_index)

        self.vocab[new_tokens[-1]] += freq

        for i in range(len(old_tokens) - 1):
            self.vocab[old_tokens[i]] -= freq
            if self.vocab[old_tokens[i]] == 0:
                del self.vocab[old_tokens[i]]

            bpe_candidate = (old_tokens[i], old_tokens[i + 1])
            bpe_token = "".join(bpe_candidate)

            pfreq = self.bpe_alignment_prob(bpe_token, freq)

            if pfreq > 0:  # just in case there is an underflow in value.
                self.merge_candidate_freq[bpe_candidate] -= pfreq
                if self.merge_candidate_freq[bpe_candidate] == 0:
                    del self.merge_candidate_freq[bpe_candidate]
                    del self.merge_candidate_indices[bpe_candidate]

        self.vocab[old_tokens[-1]] -= freq
        if self.vocab[old_tokens[-1]] == 0:
            del self.vocab[old_tokens[-1]]

    def build_vocab(
        self,
        src_txt_path: str,
        dst_txt_path: str,
        vocab_size: int,
        num_ibm_iters: int,
        num_cpus: int,
    ):
        """
        Note that except initalization, other parts are the same as the
        original bpe build_vocab method.
        """
        self._init_params(
            src_txt_path=src_txt_path,
            dst_txt_path=dst_txt_path,
            num_ibm_iters=num_ibm_iters,
            num_cpus=num_cpus,
        )
        return self._build_vocab_loop(vocab_size=vocab_size)


if __name__ == "__main__":
    arg_parser = get_arg_parser()
    options, args = arg_parser.parse_args()
    bpe_model = BilingualBPE()
    bpe_model.build_vocab(
        src_txt_path=options.src_train_file,
        dst_txt_path=options.dst_train_file,
        vocab_size=options.vocab_size,
        num_ibm_iters=options.num_ibm_iters,
        num_cpus=options.num_cpus,
    )
    bpe_model.segment_txt(
        input_path=options.src_train_file, output_path=options.train_output_file
    )
