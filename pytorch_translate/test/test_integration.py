#!/usr/bin/env python3

import contextlib
import os
import random
import tempfile
import unittest
from io import StringIO

import torch
from fairseq import options
from pytorch_translate import generate, train


class TestTranslation(unittest.TestCase):
    @unittest.skipIf(
        torch.cuda.device_count() != 1, "Test only supports single-GPU training."
    )
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
                        "256",
                        "--encoder-hidden-dim",
                        "512",
                        "--decoder-layers",
                        "2",
                        "--decoder-embed-dim",
                        "256",
                        "--decoder-hidden-dim",
                        "512",
                        "--decoder-out-embed-dim",
                        "256",
                        "--attention-type",
                        "dot",
                    ],
                )
                generate_main(data_dir)

    @unittest.skipIf(
        torch.cuda.device_count() != 1, "Test only supports single-GPU training."
    )
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
                        "256",
                        "--encoder-hidden-dim",
                        "512",
                        "--decoder-layers",
                        "2",
                        "--decoder-embed-dim",
                        "256",
                        "--decoder-hidden-dim",
                        "512",
                        "--decoder-out-embed-dim",
                        "256",
                        "--attention-type",
                        "dot",
                    ],
                )
                generate_main(data_dir)

    @unittest.skipIf(
        torch.cuda.device_count() != 1, "Test only supports single-GPU training."
    )
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
                        "64",
                        "--char-rnn-units",
                        "128",
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
                        "64",
                        "--encoder-hidden-dim",
                        "256",
                        "--decoder-layers",
                        "2",
                        "--decoder-embed-dim",
                        "256",
                        "--decoder-hidden-dim",
                        "256",
                        "--decoder-out-embed-dim",
                        "256",
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

    @unittest.skipIf(
        torch.cuda.device_count() != 1, "Test only supports single-GPU training."
    )
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
                        "256",
                        "--encoder-hidden-dim",
                        "512",
                        "--decoder-layers",
                        "2",
                        "--decoder-embed-dim",
                        "256",
                        "--decoder-hidden-dim",
                        "512",
                        "--decoder-out-embed-dim",
                        "256",
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


def write_dummy_file(filename, num_examples, maxlen):
    rng_state = torch.get_rng_state()
    torch.manual_seed(0)
    data = torch.rand(num_examples * maxlen)
    data = 97 + torch.floor(26 * data).int()
    with open(filename, "w") as h:
        offset = 0
        for _ in range(num_examples):
            ex_len = random.randint(1, maxlen)
            ex_str = " ".join(map(chr, data[offset : offset + ex_len]))
            print(ex_str, file=h)
            offset += ex_len
    torch.set_rng_state(rng_state)


def create_dummy_data(data_dir, num_examples=100, maxlen=5):
    def _create_dummy_data(filename):
        write_dummy_file(os.path.join(data_dir, filename), num_examples, maxlen)

    _create_dummy_data("train.in")
    _create_dummy_data("train.out")
    _create_dummy_data("valid.in")
    _create_dummy_data("valid.out")
    _create_dummy_data("test.in")
    _create_dummy_data("test.out")


def create_dummy_multilingual_data(data_dir, num_examples=100, maxlen=5):
    def _create_dummy_data(filename):
        write_dummy_file(os.path.join(data_dir, filename), num_examples, maxlen)

    for src, tgt in [("xh", "en"), ("zu", "en")]:
        langpair = src + tgt
        _create_dummy_data(f"train.{langpair}.{src}")
        _create_dummy_data(f"train.{langpair}.{tgt}")
        _create_dummy_data(f"tune.{langpair}.{src}")
        _create_dummy_data(f"tune.{langpair}.{tgt}")


def train_translation_model(data_dir, extra_flags):
    parser = train.get_parser_with_args()
    args = options.parse_args_and_arch(
        parser,
        [
            "--save-dir",
            data_dir,
            "--train-source-text-file",
            os.path.join(data_dir, "train.in"),
            "--train-target-text-file",
            os.path.join(data_dir, "train.out"),
            "--eval-source-text-file",
            os.path.join(data_dir, "valid.in"),
            "--eval-target-text-file",
            os.path.join(data_dir, "valid.out"),
            "--source-max-vocab-size",
            "26",
            "--target-max-vocab-size",
            "26",
            "--max-tokens",
            "500",
            "--optimizer",
            "sgd",
            "--lr",
            "0.05",
            "--lr-scheduler",
            "fixed",
            "--lr-shrink",
            "0.95",
            "--momentum",
            "0.0",
            "--clip-norm",
            "5.0",
            "--sentence-avg",
            "--criterion",
            "label_smoothed_cross_entropy",
            "--label-smoothing",
            "0.1",
            "--beam",
            "3",
            "--stop-no-best-bleu-eval",
            "5",
            "--unk-reward",
            "0.5",
            "--generate-bleu-eval-avg-checkpoints",
            "10",
            "--generate-bleu-eval-per-epoch",
            "--max-epoch",
            "1",
            "--stop-time-hr",
            "1",
            "--no-progress-bar",
            "--distributed-world-size",
            "1",
            "--source-lang",
            "in",
            "--target-lang",
            "out",
        ]
        + (extra_flags or []),
    )
    train.validate_and_set_default_args(args)
    train.main(args, train.single_process_main)


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
