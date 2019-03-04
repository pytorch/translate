#!/usr/bin/env python3

import unittest

import torch
import torch.nn.functional as F
from pytorch_translate import char_source_model  # noqa
from pytorch_translate import rnn  # noqa
from pytorch_translate.research.knowledge_distillation import (
    knowledge_distillation_loss,
)
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestKnowledgeDistillation(unittest.TestCase):
    def test_topk_kd_loss(self):
        """
        Makes sure that we can build KD loss without problem.
        """
        test_args = test_utils.ModelParamsDict()
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        self.task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        sample = self._dummy_sample()
        model = self.task.build_model(test_args)
        net_output = model(**sample["net_input"])
        student_probs = model.get_normalized_probs(net_output, log_probs=True)
        # [bsz, seqlen, vocab] -> [bsz*seqlen, vocab]
        lprobs = student_probs.view(-1, student_probs.size(-1))

        teacher_model = self.task.build_model(test_args)
        teacher_probs = teacher_model.get_normalized_probs(net_output, log_probs=False)
        top_k_teacher_probs, indices = torch.topk(teacher_probs, k=3)
        top_k_teacher_probs_normalized = F.normalize(
            top_k_teacher_probs, p=1, dim=2
        ).detach()
        sample["top_k_scores"] = top_k_teacher_probs_normalized
        sample["top_k_indices"] = indices

        kd_criterion = knowledge_distillation_loss.KnowledgeDistillationCriterion(
            test_args, self.task
        )
        kd_loss, topk_probs = kd_criterion.get_kd_loss(sample, student_probs, lprobs)

        # asserting that the values are correctly inserted into teacher_probs.
        for row in range(indices.shape[0]):
            for col in range(indices.shape[1]):
                # testing if values are normalized.
                assert round(float(torch.sum(topk_probs[row][col][:])), 0) == 1.0
                for i, val in enumerate(indices[row][col]):
                    # testing if scattering is done correctly.
                    assert (
                        topk_probs[row][col][val]
                        == top_k_teacher_probs_normalized[row][col][i]
                    )

        topk_probs_flat = topk_probs.view(-1, topk_probs.size(-1))
        kd_loss = -torch.sum(topk_probs_flat * lprobs)
        assert kd_loss >= 0

    def _dummy_sample(self):
        sample = {
            "net_input": {
                "src_tokens": torch.randint(
                    low=self.task.src_dict.nspecial,
                    high=len(self.task.src_dict),
                    size=(3, 3),
                ).long(),
                "prev_output_tokens": torch.randint(
                    low=self.task.src_dict.nspecial,
                    high=len(self.task.src_dict),
                    size=(3, 3),
                ).long(),
                "src_lengths": torch.LongTensor([3, 3, 3]),
            }
        }
        return sample
