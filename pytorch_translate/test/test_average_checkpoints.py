#!/usr/bin/env python3

import collections
import os
import tempfile
import unittest

import numpy as np
import torch
from pytorch_translate.average_checkpoints import average_checkpoints


class AverageCheckpointsTest(unittest.TestCase):
    def test_average_checkpoints(self):
        params_0 = collections.OrderedDict(
            [
                ("a", torch.DoubleTensor([100.0])),
                ("b", torch.FloatTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                ("c", torch.IntTensor([7, 8, 9])),
            ]
        )
        params_1 = collections.OrderedDict(
            [
                ("a", torch.DoubleTensor([1.0])),
                ("b", torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])),
                ("c", torch.IntTensor([2, 2, 2])),
            ]
        )
        params_avg = collections.OrderedDict(
            [
                ("a", torch.DoubleTensor([50.5])),
                ("b", torch.FloatTensor([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]])),
                # We expect truncation for integer division
                ("c", torch.IntTensor([4, 5, 5])),
            ]
        )

        fd_0, path_0 = tempfile.mkstemp()
        fd_1, path_1 = tempfile.mkstemp()
        torch.save(collections.OrderedDict([("model", params_0)]), path_0)
        torch.save(collections.OrderedDict([("model", params_1)]), path_1)

        output = average_checkpoints([path_0, path_1])["model"]

        os.close(fd_0)
        os.remove(path_0)
        os.close(fd_1)
        os.remove(path_1)

        for (k_expected, v_expected), (k_out, v_out) in zip(
            params_avg.items(), output.items()
        ):
            self.assertEquals(
                k_expected,
                k_out,
                f"Key mismatch - expected {k_expected} but found {k_out}. "
                "(Expected list of keys: {params_avg.keys()} "
                "vs actual list of keys: {output.keys()})",
            )
            np.testing.assert_allclose(
                v_expected.numpy(),
                v_out.numpy(),
                err_msg=f"Tensor value mismatch for key {k_expected}",
            )
