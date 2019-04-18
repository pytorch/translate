#!/usr/bin/env python3

import math
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, Optional, Tuple

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

        # This is the reverse of self.dst2src_ibm_model.translation_prob, but since
        # we need to search p(s|t) for only those candidates that are relevant,
        # this value is useful.
        self.dst_candidate4src_subword = {}
        for dst_word in self.dst2src_ibm_model.translation_prob.keys():
            for src_subword in self.dst2src_ibm_model.translation_prob[dst_word]:
                if src_subword not in self.dst_candidate4src_subword:
                    self.dst_candidate4src_subword[src_subword] = {}
                self.dst_candidate4src_subword[src_subword][
                    dst_word
                ] = self.dst2src_ibm_model.translation_prob[dst_word][src_subword]

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
        for target_word in self.dst2src_ibm_model.translation_prob.keys():
            top_k_pairs = sorted(
                self.dst2src_ibm_model.translation_prob[target_word].items(),
                key=lambda x: -x[1],
            )[:topk]
            denom = sum(v for (_, v) in top_k_pairs)
            self.dst2src_ibm_model.translation_prob[target_word] = defaultdict(
                float, {k: v / denom for (k, v) in top_k_pairs}
            )

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
        for (seg, freq) in self.src_bpe.current_train_data[start_index:end_index]:
            symbols = seg.split()
            for i in range(len(symbols) - 1):
                bpe_key = (symbols[i], symbols[i + 1])
                candidates[bpe_key] += freq

        for bpe_key in candidates.keys():
            prob = 0  # To avoid zero proability.
            bpe_token = "".join(bpe_key)

            if bpe_token in self.dst_candidate4src_subword:
                # p(bpe_token=c) = \sum_{t \in other_side} p(c|t) p(t)
                for target_word in self.dst_candidate4src_subword[bpe_token].keys():
                    prob += (
                        self.dst_candidate4src_subword[bpe_token][target_word]
                        * self.target_word_prob[target_word]
                    )
            candidates[bpe_key] *= max(1e-20, prob)

        return candidates

    def get_best_candidate(
        self, num_cpus: int, pool: Pool
    ) -> Optional[Tuple[str, str]]:
        """
        Calculates frequencies for new candidiates from the current vocabulary,
        and returns the candidate with the most frequency.
        """
        data_chunk_size = max(
            1, math.ceil(len(self.src_bpe.current_train_data) / num_cpus)
        )
        indices = [
            (
                i * data_chunk_size,
                min(data_chunk_size * (i + 1), len(self.src_bpe.current_train_data)),
            )
            for i in range(num_cpus)
        ]
        results = pool.map(self._best_candidate_substep, indices)
        candidates = defaultdict(float)
        for result in results:
            for (k, v) in result.items():
                candidates[k] += v
        return max(candidates, key=candidates.get) if len(candidates) > 0 else None
