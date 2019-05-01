#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import torch
from pytorch_translate.rescoring.rescorer import combine_weighted_scores
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestRescorer(unittest.TestCase):
    def test_combine_weighted_scores(self):
        test_args = test_utils.ModelParamsDict()
        test_args.enable_rescoring = True
        test_args.length_penalty = 1
        test_args.l2r_model_path = ""
        test_args.l2r_model_weight = 1.0
        test_args.r2l_model_weight = 0.0
        test_args.reverse_model_weight = 0.0
        test_args.lm_model_weight = 1.01
        test_args.length_penalty = 1.0

        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.PytorchTranslateTask(test_args, src_dict, tgt_dict)
        model = task.build_model(test_args)
        with patch(
            "pytorch_translate.utils.load_diverse_ensemble_for_inference",
            return_value=([model], test_args, task),
        ):

            scores = torch.tensor([[80, 0, 0, 0], [0, 0, 0, 80]], dtype=torch.float)
            src_tokens = torch.tensor([1, 2, 3, 4, 5])
            hypos = [{"tokens": torch.tensor([1, 2])}, {"tokens": torch.tensor([1, 2])}]

            src_len = len(src_tokens)
            tgt_len = torch.tensor(
                [len(hypo["tokens"]) for hypo in hypos], dtype=torch.float
            )
            weights = [
                test_args.l2r_model_weight,
                test_args.r2l_model_weight,
                test_args.reverse_model_weight,
                test_args.lm_model_weight,
            ]
            combined_scores = combine_weighted_scores(
                scores, weights, src_len, tgt_len, 1
            )

            # 80/(2^1), 0, 0, 80*1.01/(2^1)
            expected = torch.tensor([40.0, 40.4], dtype=torch.float)
            assert torch.equal(combined_scores, expected)
