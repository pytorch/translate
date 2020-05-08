#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
# GPU integration tests; those that don't need GPUs should go in
# test_integration.py instead

import contextlib
import tempfile
import unittest
from io import StringIO

import torch
from pytorch_translate import generate, models, train  # noqa need to load models
from pytorch_translate.test.utils import (
    create_dummy_data,
    generate_main,
    train_translation_model,
)


class TestTranslation(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 1, "Test only supports GPU training.")
    def test_rnn_fp16(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_rnn_fp16") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
                        "--fp16",
                        "--arch",
                        "rnn",
                        "--cell-type",
                        "lstm",
                        "--sequence-lstm",
                        "--reverse-source",
                        "--encoder-bidirectional",
                        "--encoder-layers",
                        "2",
                        "--encoder-embed-dim",
                        "8",
                        "--encoder-hidden-dim",
                        "16",
                        "--decoder-layers",
                        "2",
                        "--decoder-embed-dim",
                        "8",
                        "--decoder-hidden-dim",
                        "16",
                        "--decoder-out-embed-dim",
                        "8",
                        "--attention-type",
                        "dot",
                    ],
                )
                generate_main(data_dir)

    @unittest.skipIf(torch.cuda.device_count() < 1, "Test only supports GPU training.")
    def test_transformer_fp_16(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_transformer") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
                        "--fp16",
                        "--arch",
                        "ptt_transformer",
                        "--encoder-embed-dim",
                        "8",
                        "--encoder-ffn-embed-dim",
                        "16",
                        "--encoder-attention-heads",
                        "4",
                        "--encoder-layers",
                        "3",
                        "--decoder-embed-dim",
                        "8",
                        "--decoder-ffn-embed-dim",
                        "16",
                        "--decoder-attention-heads",
                        "4",
                        "--decoder-layers",
                        "3",
                    ],
                )
                generate_main(data_dir)

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Test only supports multi-GPU training."
    )
    def test_transformer_multigpu(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_transformer") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
                        "--arch",
                        "ptt_transformer",
                        "--encoder-embed-dim",
                        "256",
                        "--encoder-ffn-embed-dim",
                        "512",
                        "--encoder-attention-heads",
                        "4",
                        "--encoder-layers",
                        "3",
                        "--decoder-embed-dim",
                        "256",
                        "--decoder-ffn-embed-dim",
                        "512",
                        "--decoder-attention-heads",
                        "4",
                        "--decoder-layers",
                        "3",
                        "--distributed-world-size",
                        str(torch.cuda.device_count()),
                        "--local-num-gpus",
                        str(torch.cuda.device_count()),
                    ],
                )
                generate_main(data_dir)


if __name__ == "__main__":
    unittest.main()
