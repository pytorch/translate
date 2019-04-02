#!/usr/bin/env python3

import unittest

import torch
from pytorch_translate.rescoring.rescorer import Rescorer
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestRescorer(unittest.TestCase):
    def test_optimize(self):
        test_args = test_utils.ModelParamsDict()
        test_args.enable_rescoring = True
        test_args.original_model_weight = 1
        test_args.r2l_model_weight = 0
        test_args.reverse_model_weight = 0.5
        test_args.lm_model_weight = 0.75

        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        rescorer = Rescorer(test_args, task)

        scores = torch.tensor([[10, 20, 30, 40]], dtype=torch.float)
        src_tokens = torch.tensor([1, 2, 3, 4, 5])
        hypos = [{"tokens": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}]
        rescorer.optimize(scores, src_tokens, hypos)

        # 10=10., 20*0=0. , 30*(0.5)=15. , 40*(0.75)=30
        expected = torch.tensor([[10.0, 0.0, 15.0, 30.0]], dtype=torch.float)
        assert torch.equal(scores, expected)
