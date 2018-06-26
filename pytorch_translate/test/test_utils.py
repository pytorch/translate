#!/usr/bin/env python3

import unittest
import torch

from pytorch_translate import utils as pytorch_utils
import numpy.testing as npt
import torch.nn.functional as F


class TestAverageTensors(unittest.TestCase):
    def test_mean(self):
        a = torch.Tensor([[0.0, 2.0, 5.0], [5.0, -5.0, 6.0]])
        b = torch.Tensor([[4.0, 2.0, -1.0], [5.0, 10.0, 6.0]])
        c = torch.Tensor([[-1.0, 2.0, 5.0], [2.0, 10.0, 6.0]])
        expected = torch.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        npt.assert_allclose(pytorch_utils.average_tensors([a, b, c]), expected)
        third = 1.0 / 3.0
        npt.assert_allclose(
            pytorch_utils.average_tensors([a, b, c], weights=[third, third, third]),
            expected,
        )
        npt.assert_allclose(
            pytorch_utils.average_tensors([a, b, c], weights=[1, 1, 1]), 3 * expected
        )
        npt.assert_allclose(
            pytorch_utils.average_tensors([a, b, c], weights=[1, 0, 0]), a
        )

    def test_prob_space(self):
        a = torch.Tensor([[5.0, 5.0], [6.0, 2.0], [2.0, 6.0]])
        b = torch.Tensor([[0.0, 0.0], [2.0, 6.0], [6.0, 2.0]])
        expected = torch.Tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        npt.assert_allclose(
            pytorch_utils.average_tensors([a, b], norm_fn=F.softmax), expected
        )
