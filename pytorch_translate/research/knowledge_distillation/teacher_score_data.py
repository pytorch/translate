#!/usr/bin/env python3

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fairseq import data, utils
from pytorch_translate import utils as pytorch_translate_utils


MEM_SPLIT_SIZE = 10


class TeacherDataset(data.language_pair_dataset.LanguagePairDataset):
    """
    Extension of fairseq.data.LanguagePairDataset where each example
    has a score from a ensemble of teacher models.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        top_k_probs_binary_file: Optional[str] = None,
        teacher_models: List[Any] = None,
        top_k_teacher_scores: Dict[int, np.ndarray] = None,
        top_k_teacher_indices: Dict[int, np.ndarray] = None,
        top_k_teacher_tokens=8,
        **kwargs,
    ):
        """
        Args:
            task: The task for which the teacher models are expected to make
                prediction.
            teacher_models: A list of preloaded teacher models.
            top_k_teacher_tokens: The number of top word-level prediction that
                want to save from the teacher predictions.
            top_k_teacher_scores: A dictionary for memoization that is passed
                between the dataset and PytorchKnowledgeDistillationTask.
            top_k_teacher_scores:  A  dictionary for memoization indices of scores;
                similar to the previous argument, it is passed between the dataset
                and PytorchKnowledgeDistillationTask.
        """
        super().__init__(src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict, **kwargs)
        self.src_dict = src_dict

        self.top_k_probs_binary_file = top_k_probs_binary_file
        self.teacher_models = teacher_models
        self.top_k_teacher_tokens = top_k_teacher_tokens
        self.top_k_teacher_scores = top_k_teacher_scores
        self.top_k_teacher_indices = top_k_teacher_indices

        self.all_sen_ids_memoized = False

        if self.top_k_probs_binary_file is not None:
            npz = np.load(self.top_k_probs_binary_file)
            flat_scores = npz["top_k_scores"]
            flat_indices = npz["top_k_indices"]

            offset = 0
            for i, length in enumerate(tgt_sizes):
                scores = flat_scores[offset : offset + length, :]
                indices = flat_indices[offset : offset + length, :]
                offset += length
                self.top_k_teacher_scores[i] = torch.Tensor(scores)
                self.top_k_teacher_indices[i] = torch.LongTensor(indices)

    def __len__(self):
        return super().__len__()

    def collater(self, dataset_samples):
        return TeacherDataset.collate(
            dataset_samples,
            self.teacher_models,
            self.top_k_teacher_tokens,
            self.src_dict.pad(),
            self.src_dict.eos(),
            self.top_k_teacher_scores,
            self.top_k_teacher_indices,
        )

    @staticmethod
    def collate(
        dataset,
        teacher_models,
        top_k_teacher_tokens,
        pad_idx,
        eos_idx,
        top_k_teacher_scores: Dict[int, np.ndarray],
        top_k_teacher_indices: Dict[int, np.ndarray],
        left_pad_source=False,
    ):
        if len(dataset) == 0:
            return {}

        batched_samples = data.language_pair_dataset.collate(
            dataset, pad_idx, eos_idx, left_pad_source
        )

        sen_ids = batched_samples["id"].numpy()

        if teacher_models is not None:
            all_sen_ids_memoized = all(id in top_k_teacher_scores for id in sen_ids)

            if not all_sen_ids_memoized:
                # Because there is a high chance that the batches do not fit into memory
                # for big batches, we have to split them into smaller batches and
                # memoize their values separately.
                smaller_datasets = []

                chunk_size = max(1, math.ceil(len(dataset) / MEM_SPLIT_SIZE))
                for i in range(MEM_SPLIT_SIZE):
                    small_data = dataset[
                        chunk_size * i : min(len(dataset), (i + 1) * chunk_size)
                    ]
                    if len(small_data) > 0:
                        smaller_datasets.append(small_data)

                for smaller_data in smaller_datasets:
                    smaller_batch = data.language_pair_dataset.collate(
                        smaller_data, pad_idx, eos_idx, left_pad_source
                    )

                    sen_ids_this_batch = smaller_batch["id"].numpy()

                    # smaller_batch is natively on CPU. We want to make sure that
                    # the teacher models run on GPU.
                    net_input = {
                        key: pytorch_translate_utils.maybe_cuda(
                            smaller_batch["net_input"][key]
                        )
                        for key in smaller_batch["net_input"].keys()
                    }

                    teacher_output = teacher_models[0](**net_input)
                    avg_teacher_probs = teacher_models[0].get_normalized_probs(
                        teacher_output, log_probs=False
                    )

                    for i in range(1, len(teacher_models)):
                        teacher_output = teacher_models[i](**net_input)
                        probs = teacher_models[i].get_normalized_probs(
                            teacher_output, log_probs=False
                        )
                        avg_teacher_probs.add_(probs)

                    avg_teacher_probs.div_(len(teacher_models))
                    avg_teacher_probs = avg_teacher_probs.detach()

                    # Getting the topk probabilities, masking others,
                    # normalizing the topk probabilities.
                    top_k_teacher_tokens_avg_probs, indices = torch.topk(
                        avg_teacher_probs, k=top_k_teacher_tokens
                    )
                    top_k_teacher_probs_normalized = F.normalize(
                        top_k_teacher_tokens_avg_probs, p=1, dim=2
                    ).cpu()
                    indices = indices.cpu()

                    # Memoization
                    for id_index, id in enumerate(sen_ids_this_batch):
                        target_length = sum(
                            (batched_samples["target"][id_index] != pad_idx).numpy()
                        )
                        if id not in top_k_teacher_scores:
                            top_k_teacher_scores[id] = top_k_teacher_probs_normalized[
                                id_index
                            ][:target_length, :]
                            top_k_teacher_indices[id] = indices[id_index][
                                :target_length, :
                            ]
            else:
                # We assume that when there is a batch which is entirely memoized
                # that means we do not need the teacher models anymore, and
                # it is better to remove them from memory.
                if len(teacher_models) > 0:
                    del teacher_models[:]

        # Now we assume that all values are already memoized.
        # Preparing all zero scores and gradually filling them in.
        max_ntokens = batched_samples["target"].shape[1]
        memoized_probs = torch.zeros(len(sen_ids), max_ntokens, top_k_teacher_tokens)
        memoized_prob_idx = torch.zeros(
            len(sen_ids), max_ntokens, top_k_teacher_tokens
        ).long()

        for idx, id in enumerate(sen_ids):
            memoized_probs[idx][
                : top_k_teacher_scores[id].shape[0]
            ] = top_k_teacher_scores[id]
            memoized_prob_idx[idx][
                : top_k_teacher_indices[id].shape[0]
            ] = top_k_teacher_indices[id]
        batched_samples["top_k_scores"] = memoized_probs
        batched_samples["top_k_indices"] = memoized_prob_idx
        return batched_samples
