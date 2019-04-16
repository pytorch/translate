#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import torch
from pytorch_translate.rescoring.rescorer import Rescorer
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestRescorer(unittest.TestCase):
    def test_combine_weighted_scores(self):
        test_args = test_utils.ModelParamsDict()
        test_args.enable_rescoring = True
        test_args.length_penalty = 1
        test_args.original_model_weight = 1
        test_args.l2r_model_path = ""
        test_args.l2r_model_weight = 1
        test_args.r2l_model_weight = 0
        test_args.reverse_model_weight = 0.5
        test_args.lm_model_weight = 0.75
        test_args.length_penalty = 1

        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.PytorchTranslateTask(test_args, src_dict, tgt_dict)
        model = task.build_model(test_args)
        with patch(
            "pytorch_translate.utils.load_diverse_ensemble_for_inference",
            return_value=([model], test_args, task),
        ):
            rescorer = Rescorer(test_args)

            scores = torch.tensor([[10, 20, 30, 40]], dtype=torch.float)
            src_tokens = torch.tensor([1, 2, 3, 4, 5])
            hypos = [{"tokens": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}]
            rescorer.combine_weighted_scores(scores, src_tokens, hypos)

            # 10=1. , 20*0=0. , 30*(0.5/5)=3. , 40*(0.75/5)=6.
            expected = torch.tensor([[10.0, 0.0, 3.0, 6.0]], dtype=torch.float)
            assert torch.equal(scores, expected)
