#!/usr/bin/env python3

import unittest

import numpy as np
import torch
from pytorch_translate import char_source_model  # noqa
from pytorch_translate import rnn  # noqa
from pytorch_translate import utils as pytorch_translate_utils
from pytorch_translate.research.knowledge_distillation.teacher_score_data import (
    TeacherDataset,
)
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestTeacherScoreDataSet(unittest.TestCase):
    def test_collate(self):
        """
        Makes sure that we can memoize in collate if we give a particular data index
        in different orders.
        """
        test_args = test_utils.ModelParamsDict()
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        self.task = tasks.DictionaryHolderTask(src_dict, tgt_dict)

        teacher_model = pytorch_translate_utils.maybe_cuda(
            self.task.build_model(test_args)
        )
        teacher_models = [teacher_model]

        d0, d1, d2, d3 = self._dummy_datasets(src_dict.eos(), tgt_dict.eos())
        dataset1 = [d0, d1]
        dataset2 = [d2, d3]
        dataset3 = [d3, d0]
        dataset4 = [d1, d2]

        top_k_teacher_scores = {}
        top_k_teacher_indices = {}
        b1 = TeacherDataset.collate(
            dataset=dataset1,
            teacher_models=teacher_models,
            top_k_teacher_tokens=3,
            pad_idx=src_dict.pad(),
            eos_idx=src_dict.eos(),
            top_k_teacher_scores=top_k_teacher_scores,
            top_k_teacher_indices=top_k_teacher_indices,
            mem_split_size=3,
        )
        TeacherDataset.collate(
            dataset=dataset2,
            teacher_models=teacher_models,
            top_k_teacher_tokens=3,
            pad_idx=src_dict.pad(),
            eos_idx=src_dict.eos(),
            top_k_teacher_scores=top_k_teacher_scores,
            top_k_teacher_indices=top_k_teacher_indices,
            mem_split_size=3,
        )
        before_scores = [top_k_teacher_scores[i].cpu().numpy() for i in range(4)]
        before_indices = [top_k_teacher_indices[i].cpu().numpy() for i in range(4)]

        TeacherDataset.collate(
            dataset=dataset3,
            teacher_models=teacher_models,
            top_k_teacher_tokens=3,
            pad_idx=src_dict.pad(),
            eos_idx=src_dict.eos(),
            top_k_teacher_scores=top_k_teacher_scores,
            top_k_teacher_indices=top_k_teacher_indices,
            mem_split_size=3,
        )
        TeacherDataset.collate(
            dataset=dataset4,
            teacher_models=teacher_models,
            top_k_teacher_tokens=3,
            pad_idx=src_dict.pad(),
            eos_idx=src_dict.eos(),
            top_k_teacher_scores=top_k_teacher_scores,
            top_k_teacher_indices=top_k_teacher_indices,
            mem_split_size=3,
        )
        after_scores = [top_k_teacher_scores[i].cpu().numpy() for i in range(4)]
        after_indices = [top_k_teacher_indices[i].cpu().numpy() for i in range(4)]

        for i in range(4):
            np.array_equal(after_scores[i], before_scores[i])
            np.array_equal(after_indices[i], before_indices[i])

        b5 = TeacherDataset.collate(
            dataset=dataset1,
            teacher_models=teacher_models,
            top_k_teacher_tokens=3,
            pad_idx=src_dict.pad(),
            eos_idx=src_dict.eos(),
            top_k_teacher_scores=top_k_teacher_scores,
            top_k_teacher_indices=top_k_teacher_indices,
            mem_split_size=3,
        )

        assert len(teacher_models) == 0
        probs_before = b1["top_k_scores"].numpy()
        indices_before = b1["top_k_indices"].numpy()
        probs_after = b5["top_k_scores"].numpy()
        indices_after = b5["top_k_indices"].numpy()

        # The first one has a different length, does the last two values in the
        # before value has irrelevant values.abs
        assert np.array_equal(probs_before[0][:-4], probs_after[0][:-4]) is True
        assert np.array_equal(indices_before[0][:-4], indices_after[0][:-4]) is True
        assert np.array_equal(probs_after[0][-4:], np.zeros((4, 3))) is True
        assert np.array_equal(indices_after[0][-4:], np.zeros((4, 3))) is True

        assert np.array_equal(probs_before[1], probs_after[1]) is True
        assert np.array_equal(indices_before[1], indices_after[1]) is True

    def _dummy_datasets(self, src_eos_idx, tgt_eos_idx):
        d0 = {
            "id": 0,
            "source": torch.LongTensor(np.array([1, 9, 3, 4, 5, 6, 7, 8, src_eos_idx])),
            "target": torch.LongTensor(np.array([1, 9, 3, 4, tgt_eos_idx])),
        }
        d1 = {
            "id": 1,
            "source": torch.LongTensor(np.array([11, 9, 13, 4, 15, 6, src_eos_idx])),
            "target": torch.LongTensor(
                np.array([1, 9, 3, 4, 15, 6, 17, 18, tgt_eos_idx])
            ),
        }
        d2 = {
            "id": 2,
            "source": torch.LongTensor(
                np.array([11, 9, 3, 4, 25, 6, 7, 28, src_eos_idx])
            ),
            "target": torch.LongTensor(
                np.array([21, 9, 3, 24, 5, 6, 27, 28, tgt_eos_idx])
            ),
        }
        d3 = {
            "id": 3,
            "source": torch.LongTensor(np.array([9, 9, 9, 4, 25, 9, 7, src_eos_idx])),
            "target": torch.LongTensor(np.array([21, 9, 3, 9, 5, 9, tgt_eos_idx])),
        }
        return d0, d1, d2, d3
