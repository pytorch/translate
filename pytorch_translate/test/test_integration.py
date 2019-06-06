#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import contextlib
import os
import tempfile
import unittest
from io import StringIO

import torch
from fairseq import options
from pytorch_translate import generate, models, train  # noqa need to load models
from pytorch_translate.test.utils import (
    create_dummy_data,
    create_dummy_multilingual_data,
    train_translation_model,
)


class TestTranslation(unittest.TestCase):
    def test_rnn(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_rnn") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
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

    def test_char_rnn(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_char_rnn") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
                        "--arch",
                        "char_source",
                        "--char-embed-dim",
                        "4",
                        "--char-rnn-units",
                        "8",
                        "--char-rnn-layers",
                        "1",
                        "--char-source-max-vocab-size",
                        "26",
                        "--cell-type",
                        "lstm",
                        "--sequence-lstm",
                        "--encoder-dropout-in",
                        "0",
                        "--encoder-dropout-out",
                        "0",
                        "--decoder-dropout-in",
                        "0.2",
                        "--decoder-dropout-out",
                        "0.2",
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
                generate_main(
                    data_dir,
                    [
                        "--char-source-vocab-file",
                        os.path.join(data_dir, "char-dictionary-in.txt"),
                    ],
                )

    @unittest.skip("it's failing (T40139796)")
    def test_multilingual(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_multilingual") as data_dir:
                create_dummy_multilingual_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
                        "--task",
                        "pytorch_translate_multilingual",
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
                        "--multiling-encoder-lang",
                        "xh",
                        "--multiling-encoder-lang",
                        "zu",
                        "--multiling-encoder-lang",
                        "en",
                        "--multiling-decoder-lang",
                        "xh",
                        "--multiling-decoder-lang",
                        "en",
                        "--multiling-source-lang",
                        "xh",
                        "--multiling-target-lang",
                        "en",
                        "--multiling-train-source-text-file",
                        os.path.join(data_dir, "train.xhen.xh"),
                        "--multiling-train-target-text-file",
                        os.path.join(data_dir, "train.xhen.en"),
                        "--multiling-eval-source-text-file",
                        os.path.join(data_dir, "tune.xhen.xh"),
                        "--multiling-eval-target-text-file",
                        os.path.join(data_dir, "tune.xhen.en"),
                        "--multiling-source-lang",
                        "zu",
                        "--multiling-target-lang",
                        "en",
                        "--multiling-train-source-text-file",
                        os.path.join(data_dir, "train.zuen.zu"),
                        "--multiling-train-target-text-file",
                        os.path.join(data_dir, "train.zuen.en"),
                        "--multiling-eval-source-text-file",
                        os.path.join(data_dir, "tune.zuen.zu"),
                        "--multiling-eval-target-text-file",
                        os.path.join(data_dir, "tune.zuen.en"),
                        "--multiling-source-lang",
                        "en",
                        "--multiling-target-lang",
                        "xh",
                        "--multiling-train-source-text-file",
                        os.path.join(data_dir, "train.xhen.en"),
                        "--multiling-train-target-text-file",
                        os.path.join(data_dir, "train.xhen.xh"),
                        "--multiling-eval-source-text-file",
                        os.path.join(data_dir, "tune.xhen.en"),
                        "--multiling-eval-target-text-file",
                        os.path.join(data_dir, "tune.xhen.xh"),
                        # set these to empty to satisfy argument validation
                        "--train-source-text-file",
                        "",
                        "--train-target-text-file",
                        "",
                        "--eval-source-text-file",
                        "",
                        "--eval-target-text-file",
                        "",
                    ],
                )
                for langpair, src, tgt in [
                    ("xhen", "xh", "en"),
                    ("zuen", "zu", "en"),
                    ("xhen", "en", "xh"),
                ]:
                    generate_main(
                        data_dir,
                        [
                            "--task",
                            "pytorch_translate_multilingual",
                            "--multiling-source-lang",
                            src,
                            "--multiling-target-lang",
                            tgt,
                            "--source-vocab-file",
                            os.path.join(data_dir, f"dictionary-src-{src}.txt"),
                            "--target-vocab-file",
                            os.path.join(data_dir, f"dictionary-trg-{tgt}.txt"),
                            "--source-text-file",
                            os.path.join(data_dir, f"tune.{langpair}.{src}"),
                            "--target-text-file",
                            os.path.join(data_dir, f"tune.{langpair}.{tgt}"),
                        ],
                    )

    def test_transformer(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_transformer") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
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

    def test_char_transformer(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_char_transformer") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
                        "--arch",
                        "char_source_transformer",
                        "--char-embed-dim",
                        "4",
                        "--char-cnn-params",
                        "[(10, 1), (20, 2)]",
                        "--char-cnn-nonlinear-fn",
                        "relu",
                        "--char-cnn-num-highway-layers",
                        "2",
                        "--char-source-max-vocab-size",
                        "26",
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
                generate_main(
                    data_dir,
                    [
                        "--char-source-vocab-file",
                        os.path.join(data_dir, "char-dictionary-in.txt"),
                    ],
                )

    def test_char_source_hybrid(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_char_rnn") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
                        "--arch",
                        "char_source_hybrid",
                        "--char-embed-dim",
                        "4",
                        "--char-cnn-params",
                        "[(10, 1), (20,2)]",
                        "--char-cnn-nonlinear-fn",
                        "relu",
                        "--char-cnn-num-highway-layers",
                        "2",
                        "--char-source-max-vocab-size",
                        "26",
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
                        "--decoder-attention-heads",
                        "4",
                        "--decoder-layers",
                        "2",
                        "--decoder-lstm-units",
                        "16",
                        "--decoder-out-embed-dim",
                        "8",
                    ],
                )
                generate_main(
                    data_dir,
                    [
                        "--char-source-vocab-file",
                        os.path.join(data_dir, "char-dictionary-in.txt"),
                    ],
                )

    def test_semi_supervised_rnn(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_rnn") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
                        "--task",
                        "pytorch_translate_semi_supervised",
                        "--train-mono-source-text-file",
                        os.path.join(data_dir, "train.in"),
                        "--train-mono-target-text-file",
                        os.path.join(data_dir, "train.out"),
                        "--arch",
                        "semi_supervised_rnn",
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

    def test_semi_supervised_transformer(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_transformer") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
                        "--arch",
                        "semi_supervised_transformer",
                        # semi-supervised task args:
                        "--task",
                        "pytorch_translate_semi_supervised",
                        "--train-mono-source-text-file",
                        os.path.join(data_dir, "train.in"),
                        "--train-mono-target-text-file",
                        os.path.join(data_dir, "train.out"),
                        # transformer args:
                        "--encoder-embed-dim",
                        "4",
                        "--encoder-ffn-embed-dim",
                        "4",
                        "--encoder-attention-heads",
                        "2",
                        "--encoder-layers",
                        "1",
                        "--decoder-embed-dim",
                        "4",
                        "--decoder-ffn-embed-dim",
                        "4",
                        "--decoder-attention-heads",
                        "2",
                        "--decoder-layers",
                        "1",
                    ],
                )

    def test_denoising_autoencoder(self):
        """
        Tests denoising autoencoder task. Important flags:
        `--train-mono-*-text-file`, `--task`, `--arch`, and
        `--denoising-target-mono`.
        """
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_rnn") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
                        "--task",
                        "pytorch_translate_denoising_autoencoder",
                        "--train-mono-source-text-file",
                        os.path.join(data_dir, "train.in"),
                        "--train-mono-target-text-file",
                        os.path.join(data_dir, "train.out"),
                        "--arch",
                        "semi_supervised_rnn",
                        "--denoising-target-mono",
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

    def test_multilingual_hybrid(self):
        """
        Tests multilingual translation task. Important flags:
        `--multilingual-*-binary-path`, `--task`, `--arch`,
        `--source-vocabulary`, `--target-vocabulary`, `--vocabulary`.
        """
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_multilingual_hybrid") as data_dir:
                create_dummy_multilingual_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
                        "--task",
                        "pytorch_translate_multilingual_task",
                        "--arch",
                        "multilingual_hybrid_transformer_rnn",
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
                        "--decoder-attention-heads",
                        "4",
                        "--decoder-layers",
                        "2",
                        "--decoder-lstm-units",
                        "16",
                        "--decoder-out-embed-dim",
                        "8",
                        "--lang-pairs",
                        "xh-en,zu-en",
                        "--multilingual-train-text-file",
                        (
                            "xh-en:"
                            f"{os.path.join(data_dir, 'train.xhen.xh')},"
                            f"{os.path.join(data_dir, 'train.xhen.en')}"
                        ),
                        "--multilingual-eval-text-file",
                        (
                            "xh-en:"
                            f"{os.path.join(data_dir, 'tune.xhen.xh')},"
                            f"{os.path.join(data_dir, 'tune.xhen.en')}"
                        ),
                        "--multilingual-train-text-file",
                        (
                            "zu-en:"
                            f"{os.path.join(data_dir, 'train.zuen.zu')},"
                            f"{os.path.join(data_dir, 'train.zuen.en')}"
                        ),
                        "--multilingual-eval-text-file",
                        (
                            "zu-en:"
                            f"{os.path.join(data_dir, 'tune.zuen.zu')},"
                            f"{os.path.join(data_dir, 'tune.zuen.en')}"
                        ),
                        # set these to empty to satisfy argument validation
                        "--train-source-text-file",
                        "",
                        "--train-target-text-file",
                        "",
                        "--eval-source-text-file",
                        "",
                        "--eval-target-text-file",
                        "",
                    ],
                    # fairseq MultlilingualTranslationTask expects mandatory
                    # data directory positional argument
                    set_empty_data_positional_arg=True,
                    set_lang_args=False,
                )

    def test_word_prediction(self):
        """ Tests a word prediction model, which will use a learned vocab
        reduction via the word prediction model. It uses a custom criterion
        (word_prediction) on top of label_smoothed_cross_entropy so we pass the
        corresponding word_prediction criterion flag in during training.
        """
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_word_pred") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir=data_dir,
                    extra_flags=[
                        "--arch",
                        "rnn_word_pred",
                        "--predictor-hidden-dim",
                        "32",
                        "--topk-labels-per-source-token",
                        "5",
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
                    criterion=[
                        "--criterion",
                        "word_prediction",
                        "--label-smoothing",
                        "0.1",
                    ],
                )
                generate_main(data_dir)

    def test_pretrained_masked_lm_for_translation(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_mlm") as data_dir:
                create_dummy_data(data_dir)
                train_translation_model(
                    data_dir,
                    [
                        "--arch",
                        "xlm_base",
                        data_dir,
                        # semi-supervised task args:
                        "--task",
                        "pytorch_translate_cross_lingual_lm",
                        # transformer args:
                        "--encoder-embed-dim",
                        "4",
                        "--encoder-ffn-embed-dim",
                        "4",
                        "--encoder-attention-heads",
                        "2",
                        "--encoder-layers",
                        "1",
                        # dict files
                        "--source-vocab-file",
                        os.path.join(data_dir, "dictionary-in.txt"),
                        "--target-vocab-file",
                        os.path.join(data_dir, "dictionary-out.txt"),
                        # additoinal ones
                        "--raw-text",
                        "--monolingual-langs",
                        "in,out",
                        "--save-only",
                        "--masked-lm-only",
                        "--num-segment",
                        "2",
                    ],
                    criterion=["--criterion", "masked_lm_loss"],
                )


def generate_main(data_dir, extra_flags=None):
    parser = generate.get_parser_with_args()
    args = options.parse_args_and_arch(
        parser,
        [
            "--source-vocab-file",
            os.path.join(data_dir, "dictionary-in.txt"),
            "--target-vocab-file",
            os.path.join(data_dir, "dictionary-out.txt"),
            "--source-text-file",
            os.path.join(data_dir, "test.in"),
            "--target-text-file",
            os.path.join(data_dir, "test.out"),
            "--path",
            os.path.join(data_dir, "checkpoint_last.pt"),
            "--beam",
            "3",
            "--length-penalty",
            "0.0",
            "--batch-size",
            "64",
            "--max-len-b",
            "5",
            "--no-progress-bar",
        ]
        + (extra_flags or []),
    )
    generate.validate_args(args)
    generate.generate(args)


if __name__ == "__main__":
    unittest.main()
