#!/usr/bin/env python3

from collections import defaultdict
from typing import Dict

from pytorch_translate.research.unsupervised_morphology.bpe import BPE
from pytorch_translate.research.unsupervised_morphology.char_ibm_model1 import (
    Char2WordIBMModel1,
)


class BilingualBPE(object):
    """
    An extension of the BPE model that is cross-lingual wrt parallel data.
    """

    def __init__(self):
        self.src_bpe = BPE()
        self.dst2src_ibm_model = Char2WordIBMModel1()

    def _init_params(
        self,
        src_txt_path: str,
        dst_txt_path: str,
        num_ibm_iters: int,
        num_cpus: int,
        top_k_translations: int = 5,
    ):
        """
        Args:
            src_txt_path: Text path for source language in parallel data.
            dst_txt_path: Text path for target language in parallel data.
            num_ibm_iters: Number of training epochs for the IBM model.
            num_cpus: Number of CPUs for training the IBM model with multi-processing.
            top_k_translations: Just keep the top k in memory (for speed purposes).
        """
        self.src_bpe._init_vocab(txt_path=src_txt_path)

        # Note the reverse side of the model. Target is word based, that is why
        # we give it a reverse order.
        self.dst2src_ibm_model.learn_ibm_parameters(
            src_path=dst_txt_path,
            dst_path=src_txt_path,
            num_iters=num_ibm_iters,
            num_cpus=num_cpus,
        )
        self._prune_translation_candidates(topk=top_k_translations)

        # Probability of target words.
        self.target_word_prob = BilingualBPE.calc_word_probs(txt_path=dst_txt_path)

    @staticmethod
    def calc_word_probs(txt_path: str) -> Dict[str, float]:
        vocab = defaultdict(float)
        with open(txt_path) as txt_file:
            for line in txt_file:
                for word in line.strip().split():
                    vocab[word] += 1

        denom = sum(vocab.values())
        for word in vocab.keys():
            vocab[word] /= denom
        return vocab

    def _prune_translation_candidates(self, topk: int = 5):
        """
        Searching over all possible translation candidates is very time-consuming.
        Instead we keep a small number of top items for each candidate.
        We also normalize their value to form a probability distribution.
        """
        for bpe_type in self.dst2src_ibm_model.translation_prob.keys():
            top_k_pairs = sorted(
                self.dst2src_ibm_model.translation_prob[bpe_type].items(),
                key=lambda x: -x[1],
            )[:topk]
            denom = sum(v for (_, v) in top_k_pairs)
            self.dst2src_ibm_model.translation_prob[bpe_type] = defaultdict(
                float, {k: v / denom for (k, v) in top_k_pairs}
            )
