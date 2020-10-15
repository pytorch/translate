#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import torch
from pytorch_translate.rescoring.model_scorers import (
    R2LModelScorer,
    ReverseModelScorer,
    SimpleModelScorer,
)
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestModelScorers(unittest.TestCase):
    def setUp(self):
        self.args = test_utils.ModelParamsDict()
        _, src_dict, tgt_dict = test_utils.prepare_inputs(self.args)
        self.task = tasks.PytorchTranslateTask(self.args, src_dict, tgt_dict)
        self.model = self.task.build_model(self.args)

    def test_reverse_tgt_tokens(self):
        with patch(
            "pytorch_translate.utils.load_diverse_ensemble_for_inference",
            return_value=([self.model], self.args, self.task),
        ):
            scorer = R2LModelScorer(self.args, "/tmp/model_path.txt")
            pad = self.task.tgt_dict.pad()
            tgt_tokens = torch.Tensor([[1, 2, 3], [1, 2, pad], [1, pad, pad]])
            expected_tokens = torch.Tensor([[3, 2, 1], [2, 1, pad], [1, pad, pad]])
            reversed_tgt_tokens = scorer.reverse_tgt_tokens(tgt_tokens)
            assert torch.equal(reversed_tgt_tokens, expected_tokens)

    def test_convert_hypos_to_tgt_tokens(self):
        with patch(
            "pytorch_translate.utils.load_diverse_ensemble_for_inference",
            return_value=([self.model], self.args, self.task),
        ):
            scorer = SimpleModelScorer(self.args, "/tmp/model_path.txt")
            hypos = [
                {"tokens": torch.Tensor([1, 2, 3, 4, 5])},
                {"tokens": torch.Tensor([1, 2, 3, 4])},
                {"tokens": torch.Tensor([1, 2, 3])},
                {"tokens": torch.Tensor([1, 2])},
                {"tokens": torch.Tensor([1])},
            ]
            tgt_tokens = scorer.convert_hypos_to_tgt_tokens(hypos)

            pad = self.task.tgt_dict.pad()
            eos = self.task.tgt_dict.eos()
            expected_tgt_tokens = torch.Tensor(
                [
                    [eos, 1, 2, 3, 4, 5],
                    [eos, 1, 2, 3, 4, pad],
                    [eos, 1, 2, 3, pad, pad],
                    [eos, 1, 2, pad, pad, pad],
                    [eos, 1, pad, pad, pad, pad],
                ]
            ).type_as(tgt_tokens)
            assert torch.equal(tgt_tokens, expected_tgt_tokens)

    def test_compute_scores(self):
        # TODO(halilakin): Verify behaviour in batch mode
        with patch(
            "pytorch_translate.utils.load_diverse_ensemble_for_inference",
            return_value=([self.model], self.args, self.task),
        ):
            scorer = SimpleModelScorer(self.args, "/tmp/model_path.txt")
            tgt_tokens = torch.tensor([[2, 11, 22, 0], [2, 33, 44, 55]])
            logprobs = torch.zeros(
                tgt_tokens.shape[0], tgt_tokens.shape[1], len(self.task.tgt_dict)
            )
            logprobs[0, 0, 11] = 0.5
            logprobs[0, 1, 22] = 1.5
            logprobs[0, 3, :] = 5

            logprobs[1, 0, 33] = 0.5
            logprobs[1, 1, 44] = 1.5
            logprobs[1, 2, 55] = 2.5

            hypos_scores = scorer.compute_scores(tgt_tokens, logprobs)
            assert hypos_scores[0] == 2.0
            assert hypos_scores[1] == 4.5

    def test_simple_scorer_prepare_inputs(self):
        pad = self.task.tgt_dict.pad()
        eos = self.task.tgt_dict.eos()

        src_tokens = torch.tensor([[6, 7, 8]], dtype=torch.int)
        hypos = [
            {"tokens": torch.tensor([12, 13, 14, eos], dtype=torch.int)},
            {"tokens": torch.tensor([22, 23, eos], dtype=torch.int)},
        ]

        with patch(
            "pytorch_translate.utils.load_diverse_ensemble_for_inference",
            return_value=([self.model], self.args, self.task),
        ):
            scorer = SimpleModelScorer(
                self.args, "/tmp/model_path.txt", None, self.task
            )
            (encoder_inputs, tgt_tokens) = scorer.prepare_inputs(src_tokens, hypos)

            # Test encoder inputs
            assert torch.equal(
                encoder_inputs[0], torch.tensor([[6, 7, 8], [6, 7, 8]], dtype=torch.int)
            ), "Encoder inputs are not as expected"
            assert torch.equal(
                encoder_inputs[1], torch.tensor([3, 3], dtype=torch.int)
            ), "Src lengths are not as expected"

            # Test target tokens
            assert torch.equal(
                tgt_tokens,
                torch.tensor(
                    [[eos, 12, 13, 14, eos], [eos, 22, 23, eos, pad]], dtype=torch.int
                ),
            ), "Target tokens are not as expected"

    def test_reverse_scorer_prepare_inputs(self):
        self.args.append_eos_to_source = True
        pad = self.task.tgt_dict.pad()
        eos = self.task.tgt_dict.eos()

        src_tokens = torch.tensor([6, 7, 8], dtype=torch.int)
        hypos = [
            {"tokens": torch.tensor([12, 13, 14, eos], dtype=torch.int)},
            {"tokens": torch.tensor([22, 23, eos], dtype=torch.int)},
        ]

        with patch(
            "pytorch_translate.utils.load_diverse_ensemble_for_inference",
            return_value=([self.model], self.args, self.task),
        ):
            scorer = ReverseModelScorer(
                self.args, "/tmp/model_path.txt", None, self.task
            )
            (encoder_inputs, tgt_tokens) = scorer.prepare_inputs(src_tokens, hypos)

            # Test encoder inputs
            assert torch.equal(
                encoder_inputs[0],
                torch.tensor([[12, 13, 14, eos], [22, 23, eos, pad]], dtype=torch.int),
            ), "Encoder inputs are not as expected"
            max_tgt_len = max(len(hypo["tokens"]) for hypo in hypos)
            assert encoder_inputs[1][0] == max_tgt_len, " Src length is not as expected"

            # Test target tokens
            assert torch.equal(
                tgt_tokens,
                torch.tensor(
                    [[eos, 6, 7, 8, eos], [eos, 6, 7, 8, eos]], dtype=torch.int
                ),
            ), "Target tokens are not as expected"

            # Turn off append_eos_to_source and verify outputs again
            self.args.append_eos_to_source = False
            (encoder_inputs, tgt_tokens) = scorer.prepare_inputs(src_tokens, hypos)

            # Test encoder inputs
            assert torch.equal(
                encoder_inputs[0],
                torch.tensor([[12, 13, 14], [22, 23, pad]], dtype=torch.int),
            ), "Encoder inputs are not as expected"
            max_tgt_len = max(len(hypo["tokens"]) for hypo in hypos)
            assert (
                encoder_inputs[1][0] == max_tgt_len - 1
            ), " Src length is not as expected"

            # Test target tokens
            assert torch.equal(
                tgt_tokens,
                torch.tensor(
                    [[eos, 6, 7, 8, eos], [eos, 6, 7, 8, eos]], dtype=torch.int
                ),
            ), "Target tokens are not as expected"

    def test_r2l_scorer_prepare_inputs(self):
        eos = self.task.tgt_dict.eos()
        src_tokens = torch.tensor([[6, 7, 8], [1, 2, 3]], dtype=torch.int)
        hypos = [
            {"tokens": torch.tensor([12, 13, 14, eos], dtype=torch.int)},
            {"tokens": torch.tensor([22, 23, eos], dtype=torch.int)},
            {"tokens": torch.tensor([12, 13, 14, eos], dtype=torch.int)},
            {"tokens": torch.tensor([22, 23, eos], dtype=torch.int)},
        ]

        with patch(
            "pytorch_translate.utils.load_diverse_ensemble_for_inference",
            return_value=([self.model], self.args, self.task),
        ):
            scorer = R2LModelScorer(self.args, "/tmp/model_path.txt", None, self.task)
            (encoder_inputs, tgt_tokens) = scorer.prepare_inputs(src_tokens, hypos)
            # Test encoder inputs
            assert torch.equal(
                encoder_inputs[0],
                torch.tensor(
                    [[6, 7, 8], [6, 7, 8], [1, 2, 3], [1, 2, 3]], dtype=torch.int
                ),
            ), "Encoder inputs are not as expected"

    def test_padding(self):
        """Same sentence should produce the same score with or without padding"""
        eos = self.task.tgt_dict.eos()

        src_tokens = torch.tensor([[6, 7, 8]], dtype=torch.long)
        hypos_with_padding = [
            {"tokens": torch.tensor([12, 13, 14, 15, 16, eos], dtype=torch.long)},
            {"tokens": torch.tensor([22, 23, 24, 25, eos], dtype=torch.long)},
            {"tokens": torch.tensor([32, 33, eos], dtype=torch.long)},
        ]
        hypos_without_padding = [
            {"tokens": torch.tensor([22, 23, 24, 25, eos], dtype=torch.long)},
            {"tokens": torch.tensor([32, 33, eos], dtype=torch.long)},
        ]

        with patch(
            "pytorch_translate.utils.load_diverse_ensemble_for_inference",
            return_value=([self.model], self.args, self.task),
        ):
            scorer = SimpleModelScorer(
                self.args, "/tmp/model_path.txt", None, self.task
            )
            scores_with_padding = scorer.score(src_tokens, hypos_with_padding)
            scores_without_padding = scorer.score(src_tokens, hypos_without_padding)
            assert (
                scores_with_padding[1] == scores_without_padding[0]
                and scores_with_padding[2] == scores_without_padding[1]
            ), "Scores with and without padding should not be different"

    def test_reverse_model_scorer(self):
        """Verify that reverse model is working correctly, by having one
        forward and one backward scorers, and asserting that we get the same
        scores from two scorers when source and targets are reversed

        """
        eos = self.task.tgt_dict.eos()

        src_tokens = torch.tensor([[6, 7, 8]], dtype=torch.long)
        hypos = [
            {"tokens": torch.tensor([12, 13, 14, 15, eos], dtype=torch.long)},
            {"tokens": torch.tensor([22, 23, 24, eos], dtype=torch.long)},
            {"tokens": torch.tensor([32, 33, eos], dtype=torch.long)},
        ]

        reverse_src_tokens_0 = torch.tensor([[12, 13, 14, 15]], dtype=torch.long)
        reverse_src_tokens_1 = torch.tensor([[22, 23, 24]], dtype=torch.long)
        reverse_src_tokens_2 = torch.tensor([[32, 33]], dtype=torch.long)
        reverse_hypos = [{"tokens": torch.tensor([6, 7, 8, eos], dtype=torch.long)}]

        with patch(
            "pytorch_translate.utils.load_diverse_ensemble_for_inference",
            return_value=([self.model], self.args, self.task),
        ):
            scorer = SimpleModelScorer(
                self.args, "/tmp/model_path.txt", None, self.task
            )
            reverse_scorer = ReverseModelScorer(
                self.args, "/tmp/model_path.txt", None, self.task
            )
            forward_scores = scorer.score(src_tokens, hypos)
            reverse_score_0 = reverse_scorer.score(reverse_src_tokens_0, reverse_hypos)
            reverse_score_1 = reverse_scorer.score(reverse_src_tokens_1, reverse_hypos)
            reverse_score_2 = reverse_scorer.score(reverse_src_tokens_2, reverse_hypos)

            assert forward_scores[0] == reverse_score_0[0]
            assert forward_scores[1] == reverse_score_1[0]
            assert forward_scores[2] == reverse_score_2[0]
