#!/usr/bin/env python3

import unittest

import torch
from pytorch_translate import char_source_model  # noqa
from pytorch_translate import rnn  # noqa
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
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # [bsz, seqlen, vocab] -> [bsz*seqlen, vocab]
        lprobs = lprobs.view(-1, lprobs.size(-1))

        teacher_model = self.task.build_model(test_args)
        teacher_probs = teacher_model.get_normalized_probs(net_output, log_probs=False)
        teacher_probs = teacher_probs.view(-1, teacher_probs.size(-1)).detach()

        # Getting the topk probabilities, masking others, normalizing the topk
        # probabilities.
        top_k_probs, indices = torch.topk(teacher_probs, k=3)
        top_k_probs_normalized = top_k_probs / torch.sum(top_k_probs)
        topk_mask = torch.zeros(teacher_probs.shape)
        topk_probs = topk_mask.scatter(1, indices, top_k_probs_normalized)

        # asserting that the values are correctly asserted into teacher_probs.
        for i, index in enumerate(indices):
            print(index, i)
            for j in index:
                if teacher_probs[i][j] > 0:
                    assert topk_probs[i][j] > 0

        kd_loss = -torch.sum(topk_probs * lprobs)
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
