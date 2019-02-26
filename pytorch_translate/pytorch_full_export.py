#!/usr/bin/env python3

import argparse

from pytorch_translate import rnn  # noqa
from pytorch_translate.ensemble_export import BeamSearch


def main():
    parser = argparse.ArgumentParser(
        description=("Export PyTorch-trained FBTranslate models to caffe2")
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs="+",
        help="PyTorch checkpoint file (at least one required)",
    )
    parser.add_argument(
        "--output_file",
        default="",
        help="File name to which to save beam search network",
    )
    parser.add_argument(
        "--src_dict",
        required=True,
        help="File encoding PyTorch dictionary for source language",
    )
    parser.add_argument(
        "--dst_dict",
        required=True,
        help="File encoding PyTorch dictionary for source language",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=6,
        help="Number of top candidates returned by each decoder step",
    )
    parser.add_argument(
        "--word_reward",
        type=float,
        default=0.0,
        help="Value to add for each word (besides EOS)",
    )
    parser.add_argument(
        "--unk_reward",
        type=float,
        default=0.0,
        help="Value to add for each word UNK token",
    )

    args = parser.parse_args()

    if args.output_file == "":
        print("No action taken. Need output_file to be specified.")
        parser.print_help()
        return

    checkpoint_filenames = [arg[0] for arg in args.checkpoint]

    beam_search = BeamSearch.build_from_checkpoints(
        checkpoint_filenames=checkpoint_filenames,
        src_dict_filename=args.src_dict,
        dst_dict_filename=args.dst_dict,
        beam_size=args.beam_size,
        word_reward=args.word_reward,
        unk_reward=args.unk_reward,
    )
    beam_search.save_to_pytorch(args.output_file)


if __name__ == "__main__":
    main()
