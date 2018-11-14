#!/usr/bin/env python3

import argparse

import numpy as np
import torch
from pytorch_translate import rnn  # noqa
from pytorch_translate.ensemble_export import (
    CharSourceEncoderEnsemble,
    DecoderBatchedStepEnsemble,
    EncoderEnsemble,
)


def get_parser_with_args():
    parser = argparse.ArgumentParser(
        description=("Export PyTorch-trained FBTranslate models to Caffe2 components")
    )

    parser.add_argument(
        "--path",
        "--checkpoint",
        metavar="FILE",
        help="path(s) to model file(s), colon separated",
    )
    parser.add_argument(
        "--encoder-output-file",
        default="",
        help="File name to which to save encoder ensemble network",
    )
    parser.add_argument(
        "--decoder-output-file",
        default="",
        help="File name to which to save decoder step ensemble network",
    )
    parser.add_argument(
        "--source-vocab-file",
        required=True,
        help="File encoding PyTorch dictionary for source language",
    )
    parser.add_argument(
        "--target-vocab-file",
        required=True,
        help="File encoding PyTorch dictionary for source language",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=6,
        help="Number of top candidates returned by each decoder step",
    )
    parser.add_argument(
        "--word-reward",
        type=float,
        default=0.0,
        help="Value to add for each word (besides EOS)",
    )
    parser.add_argument(
        "--unk-reward",
        type=float,
        default=0.0,
        help="Value to add for each word UNK token",
    )
    parser.add_argument(
        "--char-source",
        action="store_true",
        help=(
            "Indicates encoder uses char_source architecture (taking both "
            "token and character numberized inputs)."
        ),
    )

    return parser


def assert_required_args_are_set(args):
    if args.encoder_output_file == args.decoder_output_file == "":
        print(
            "No action taken. Need at least one of --encoder-output-file "
            "and --decoder-output-file."
        )
        return


def main():
    parser = get_parser_with_args()
    args = parser.parse_args()
    export(args)


def export(args):
    assert_required_args_are_set(args)
    checkpoint_filenames = args.path.split(":")

    if args.char_source:
        encoder_class = CharSourceEncoderEnsemble
    else:
        encoder_class = EncoderEnsemble

    encoder_ensemble = encoder_class.build_from_checkpoints(
        checkpoint_filenames=checkpoint_filenames,
        src_dict_filename=args.source_vocab_file,
        dst_dict_filename=args.target_vocab_file,
    )
    if args.encoder_output_file != "":
        encoder_ensemble.save_to_db(args.encoder_output_file)

    if args.decoder_output_file != "":
        decoder_step_ensemble = DecoderBatchedStepEnsemble.build_from_checkpoints(
            checkpoint_filenames=checkpoint_filenames,
            src_dict_filename=args.source_vocab_file,
            dst_dict_filename=args.target_vocab_file,
            beam_size=args.beam_size,
            word_reward=args.word_reward,
            unk_reward=args.unk_reward,
        )

        # need example encoder outputs to pass through network
        # (source length 5 is arbitrary)
        src_dict = encoder_ensemble.src_dict
        token_list = [src_dict.unk()] * 4 + [src_dict.eos()]
        src_tokens = torch.LongTensor(
            np.array(token_list, dtype="int64").reshape(-1, 1)
        )
        src_lengths = torch.IntTensor(np.array([len(token_list)], dtype="int32"))
        if args.char_source:
            char_inds = torch.LongTensor(np.ones((1, 5, 3), dtype="int64"))
            word_lengths = torch.LongTensor(np.array([3] * 5, dtype="int64")).reshape(
                1, 5
            )
            pytorch_encoder_outputs = encoder_ensemble(
                src_tokens, src_lengths, char_inds, word_lengths
            )
        else:
            pytorch_encoder_outputs = encoder_ensemble(src_tokens, src_lengths)

        decoder_step_ensemble.save_to_db(
            args.decoder_output_file, pytorch_encoder_outputs
        )


if __name__ == "__main__":
    main()
