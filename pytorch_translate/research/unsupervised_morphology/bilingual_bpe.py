#!/usr/bin/env python3

import datetime
import math
from collections import defaultdict
from multiprocessing import Pool
from optparse import OptionParser
from typing import Dict, Optional, Tuple

from pytorch_translate.research.unsupervised_morphology.bpe import BPE
from pytorch_translate.research.unsupervised_morphology.char_ibm_model1 import (
    CharIBMModel1,
)


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
        "--src-vocab-size",
        type="int",
        dest="src_vocab_size",
        help="Source vocabulary Size.",
        default=20000,
    )
    parser.add_option(
        "--dst-vocab-size",
        type="int",
        dest="dst_vocab_size",
        help="Target vocabulary Size.",
        default=20000,
    )
    parser.add_option(
        "--src-train-out",
        dest="src_train_output_file",
        help="BPE tokenized source train file.",
        metavar="FILE",
        default=None,
    )
    parser.add_option(
        "--dst-train-out",
        dest="dst_train_output_file",
        help="BPE tokenized target train file.",
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
        self,
        src_txt_path: str,
        dst_txt_path: str,
        num_ibm_iters: int,
        num_cpus: int,
        topk_translation_candidates: int = 5,
    ):
        """
        Args:
            src_txt_path: Text path for source language in parallel data.
            dst_txt_path: Text path for target language in parallel data.
            num_ibm_iters: Number of training epochs for the IBM model.
            num_cpus: Number of CPUs for training the IBM model with multi-processing.
            topk_translation_candidates: Used for getting top-k translaiton
                candidates after every step in BPE merge.
        """
        print(str(datetime.datetime.now()), "initializing the source vocab")
        self.src_bpe._init_vocab(txt_path=src_txt_path)
        print(str(datetime.datetime.now()), "initializing the target vocab")
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
        self._prune_translation_candidates(
            topk=topk_translation_candidates, for_src=True
        )
        self._prune_translation_candidates(
            topk=topk_translation_candidates, for_src=False
        )

    def _prune_translation_candidates(self, for_src: bool, topk: int = 5):
        """
        Searching over all possible translation candidates is very time-consuming.
        Instead we keep a small number of top items for each candidate.
        We also normalize their value to form a probability distribution.
        """
        if for_src:
            self.src_pruned_translation = defaultdict()
        else:
            self.dst_pruned_translation = defaultdict()

        to_prune = (
            self.src_pruned_translation if for_src else self.dst_pruned_translation
        )
        t_dict = (
            self.src2dst_ibm_model.translation_prob
            if for_src
            else self.dst2src_ibm_model.translation_prob
        )
        vocab = self.dst_bpe.vocab if for_src else self.src_bpe.vocab
        for key in t_dict.keys():
            pruned_vocab = {k: v for (k, v) in t_dict[key].items() if k in vocab}
            top_k_pairs = sorted(pruned_vocab.items(), key=lambda x: -x[1])[:topk]
            denom = sum(v for (_, v) in top_k_pairs)
            to_prune[key] = defaultdict(float, {k: v / denom for (k, v) in top_k_pairs})

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
        translation_probs = (
            self.dst_pruned_translation if for_src else self.src_pruned_translation
        )

        start_index, end_index = start_end_indices[0], start_end_indices[1]
        assert start_index <= end_index

        candidates = defaultdict(float)
        for (seg, freq) in bpe_model.current_train_data[start_index:end_index]:
            symbols = seg.split()
            for i in range(len(symbols) - 1):
                bpe_key = (symbols[i], symbols[i + 1])
                bpe_token = "".join([symbols[i], symbols[i + 1]])
                prob = 0

                candidates[bpe_key] += freq

        for bpe_key in candidates.keys():
            bpe_prob = 0
            # p(bpe_type=c) = \sum_{t \in other_side} p(c|t) p(t)
            for other_side_bpe_type in translation_probs.keys():
                translation_prob = (
                    translation_probs[bpe_token][other_side_bpe_type]
                    if bpe_token in translation_probs
                    and other_side_bpe_type in other_side_bpe_model.vocab
                    else 1e-10
                )
                bpe_prob += (
                    other_side_bpe_model.vocab[other_side_bpe_type] * translation_prob
                )
            candidates[bpe_key] *= bpe_prob

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

    def build_vocab(
        self,
        src_txt_path: str,
        dst_txt_path: str,
        src_vocab_size: int,
        dst_vocab_size: int,
        num_ibm_iters: int,
        num_cpus: int,
        topk_translation_candidates: int = 5,
    ):
        """
        Args:
            topk_translation_candidates: The number of top candidates to keep in
                subword IBM model translation candidates.
        """
        self._init_params(
            src_txt_path=src_txt_path,
            dst_txt_path=dst_txt_path,
            num_ibm_iters=num_ibm_iters,
            num_cpus=num_cpus,
            topk_translation_candidates=topk_translation_candidates,
        )

        src_vocab_finished = len(self.src_bpe.vocab) >= src_vocab_size
        dst_vocab_finished = len(self.dst_bpe.vocab) >= dst_vocab_size

        with Pool(processes=num_cpus) as pool:
            step = 0
            while not (src_vocab_finished and dst_vocab_finished):
                if not src_vocab_finished:
                    src_merge_candidate = self.get_best_candidate(
                        num_cpus=num_cpus, pool=pool, for_src=True
                    )
                    if src_merge_candidate is None:
                        src_vocab_finished = True
                    else:
                        self.src_bpe.merge_candidate_into_vocab(
                            candidate=src_merge_candidate, num_cpus=num_cpus, pool=pool
                        )
                        src_vocab_finished = len(self.src_bpe.vocab) >= src_vocab_size
                        # Prune the dictionary for the target side with respect
                        # to the updated entries in the source side.
                        self._prune_translation_candidates(
                            topk=topk_translation_candidates, for_src=False
                        )

                if not dst_vocab_finished:
                    dst_merge_candidate = self.get_best_candidate(
                        num_cpus=num_cpus, pool=pool, for_src=False
                    )
                    if dst_merge_candidate is None:
                        dst_vocab_finished = True
                    else:
                        self.dst_bpe.merge_candidate_into_vocab(
                            candidate=dst_merge_candidate, num_cpus=num_cpus, pool=pool
                        )
                        dst_vocab_finished = len(self.dst_bpe.vocab) >= dst_vocab_size
                        # Prune the dictionary for the source side with respect to the
                        # updated entries in the target side.
                        self._prune_translation_candidates(
                            topk=topk_translation_candidates, for_src=True
                        )

            step += 1
            if step % 1 == 0:
                print(
                    str(datetime.datetime.now()),
                    "BPE merging step",
                    step,
                    "current vocabulary size",
                    len(self.src_bpe.vocab),
                    len(self.dst_bpe.vocab),
                )


if __name__ == "__main__":
    arg_parser = get_arg_parser()
    options, args = arg_parser.parse_args()
    bpe_model = BilingualBPE()
    bpe_model.build_vocab(
        src_txt_path=options.src_train_file,
        dst_txt_path=options.dst_train_file,
        src_vocab_size=options.dst_vocab_size,
        dst_vocab_size=options.dst_vocab_size,
        num_ibm_iters=options.num_ibm_iters,
        num_cpus=options.num_cpus,
    )
    bpe_model.src_bpe.segment_txt(
        input_path=options.src_train_file, output_path=options.src_train_output_file
    )
    bpe_model.dst_bpe.segment_txt(
        input_path=options.dst_train_file, output_path=options.dst_train_output_file
    )
