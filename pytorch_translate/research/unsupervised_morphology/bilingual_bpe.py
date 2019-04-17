#!/usr/bin/env python3

import math
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, Optional, Tuple

from pytorch_translate.research.unsupervised_morphology.bpe import BPE
from pytorch_translate.research.unsupervised_morphology.char_ibm_model1 import (
    CharIBMModel1,
)


class BilingualBPE(object):
    """
    An extension of the BPE model that is cross-lingual wrt parallel data.
    """

    def __init__(self):
        self.src_bpe = BPE()
        self.dst_bpe = BPE()
        self.src2dst_ibm_model = CharIBMModel1()
        self.dst2src_ibm_model = CharIBMModel1()

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
        self.src_bpe._init_vocab(txt_path=src_txt_path)
        self.dst_bpe._init_vocab(txt_path=dst_txt_path)
        self.src2dst_ibm_model.learn_ibm_parameters(
            src_path=src_txt_path,
            dst_path=dst_txt_path,
            num_iters=num_ibm_iters,
            num_cpus=num_cpus,
        )
        self.dst2src_ibm_model.learn_ibm_parameters(
            src_path=dst_txt_path,
            dst_path=src_txt_path,
            num_iters=num_ibm_iters,
            num_cpus=num_cpus,
        )

    def _best_candidate_substep(
        self, start_end_indices: Tuple[int, int], for_src: bool
    ) -> Dict[Tuple[str, str], float]:
        """
        Args:
            start_end_indices: first and end index for part of
                self.current_train_data to search for.
            for_src: finding the best candidate for src or target.
        """
        bpe_model = self.src_bpe if for_src else self.dst_bpe
        other_side_bpe_model = self.dst_bpe if for_src else self.src_bpe
        ibm_model = self.src2dst_ibm_model if for_src else self.dst2src_ibm_model

        start_index, end_index = start_end_indices[0], start_end_indices[1]
        assert start_index <= end_index

        candidates = defaultdict(float)
        for (seg, freq) in bpe_model.current_train_data[start_index:end_index]:
            symbols = seg.split()
            for i in range(len(symbols) - 1):
                bpe_key = (symbols[i], symbols[i + 1])
                bpe_token = "".join([symbols[i], symbols[i + 1]])
                prob = 0

                # p(bpe_type=c) = \sum_{t \in other_side} p(c|t) p(t)
                for other_side_bpe_type in other_side_bpe_model.vocab.keys():
                    translation_prob = (
                        ibm_model.translation_prob[bpe_token][other_side_bpe_type]
                        if bpe_token in ibm_model.translation_prob
                        else 1e-10
                    )
                    prob += freq * (
                        other_side_bpe_model.vocab[other_side_bpe_type]
                        * translation_prob
                    )
                candidates[bpe_key] += prob

        return candidates

    def get_best_candidate(
        self, num_cpus: int, pool: Pool, for_src: bool
    ) -> Optional[Tuple[str, str]]:
        """
        Calculates frequencies for new candidiates from the current vocabulary,
        and returns the candidate with the most frequency.
        Args:
            for_src: finding the best candidate for src or target.
        """
        data_chunk_size = max(
            1, math.ceil(len(self.src_bpe.current_train_data) / num_cpus)
        )
        indices = [
            (
                (
                    i * data_chunk_size,
                    min(
                        data_chunk_size * (i + 1), len(self.src_bpe.current_train_data)
                    ),
                ),
                for_src,
            )
            for i in range(num_cpus)
        ]
        results = pool.starmap(self._best_candidate_substep, indices)
        candidates = defaultdict(float)
        for result in results:
            for (k, v) in result.items():
                candidates[k] += v
        return max(candidates, key=candidates.get) if len(candidates) > 0 else None
