#!/usr/bin/env python3

import unittest

import numpy.testing as npt
import torch
import torch.nn.functional as F
from pytorch_translate import utils as pytorch_utils


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


class TestMaybeCat(unittest.TestCase):
    def test_cat(self):
        a = torch.IntTensor([[1, 2, 3], [4, 5, 6]])
        b = torch.IntTensor([[11, 12, 13], [14, 15, 16]])
        ab = torch.IntTensor([[1, 2, 3, 11, 12, 13], [4, 5, 6, 14, 15, 16]])
        npt.assert_array_equal(pytorch_utils.maybe_cat([a, b], dim=1), ab)
        npt.assert_array_equal(
            pytorch_utils.maybe_cat([a, None, b, None, None], dim=1), ab
        )
        npt.assert_array_equal(pytorch_utils.maybe_cat([None, None, a, None], dim=1), a)

    def test_nullable(self):
        a = torch.IntTensor([[1, 2, 3], [4, 5, 6]])
        pytorch_utils.maybe_cat([a, None], 1)
        pytorch_utils.maybe_cat([a, None], 1, nullable=[True, True])
        pytorch_utils.maybe_cat([a, None], 1, nullable=[False, True])
        with self.assertRaises(RuntimeError):
            pytorch_utils.maybe_cat([a, None], 1, nullable=[False, False])
        with self.assertRaises(RuntimeError):
            pytorch_utils.maybe_cat([None, None], 1)
        with self.assertRaises(RuntimeError):
            pytorch_utils.maybe_cat([], 1)
