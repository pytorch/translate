#!/usr/bin/env python3

from collections import defaultdict
from typing import Dict, Tuple

from pytorch_translate.research.unsupervised_morphology.bpe import BPE
from pytorch_translate.research.unsupervised_morphology.char_ibm_model1 import (
    Word2CharIBMModel1,
)


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
        self._init_vocab(txt_path=src_txt_path)

        # Note the reverse side of the model. Target is word based, that is why
        # we give it a reverse order.
        self.dst2src_ibm_model.learn_ibm_parameters(
            src_path=dst_txt_path,
            dst_path=src_txt_path,
            num_iters=num_ibm_iters,
            num_cpus=num_cpus,
        )

        self.bpe_probs_from_alignment = self._calc_bpe_prob_from_alignment(
            dst_txt_path=dst_txt_path
        )

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

    def _calc_bpe_prob_from_alignment(self, dst_txt_path) -> Dict[str, float]:
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
        return bpe_alignment_prob

    def _best_candidate_substep(
        self, start_end_indices: Tuple[int, int]
    ) -> Dict[Tuple[str, str], float]:
        """
        Args:
            start_end_indices: first and end index for part of
                self.current_train_data to search for.
        """

        start_index, end_index = start_end_indices[0], start_end_indices[1]
        assert start_index <= end_index

        candidates = defaultdict(float)
        for (seg, freq) in self.current_train_data[start_index:end_index]:
            symbols = seg.split()
            for i in range(len(symbols) - 1):
                bpe_key = (symbols[i], symbols[i + 1])
                bpe_token = "".join(bpe_key)

                # Note that this line is the only difference between this class
                # and its parent BPE.
                candidates[bpe_key] += freq * self.bpe_probs_from_alignment[bpe_token]
        return candidates
