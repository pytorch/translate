#!/usr/bin/env python3

import argparse

from pytorch_translate import rnn  # noqa
from pytorch_translate.constants import CHECKPOINT_PATHS_DELIMITER
from pytorch_translate.ensemble_export import BeamSearch


def get_parser_with_args():
    parser = argparse.ArgumentParser(
        description=("Export PyTorch-trained FBTranslate models")
    )
    parser.add_argument(
        "--path",
        "--checkpoint",
        metavar="FILE",
        help="path(s) to model file(s), colon separated",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="File name to which to save beam search network",
    )
    parser.add_argument(
        "--output-graph-file",
        default="",
        help="File name to which to save the beam search graph for debugging",
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

    return parser


def main():
    parser = get_parser_with_args()
    args = parser.parse_args()

    if args.output_file == "":
        print("No action taken. Need output_file to be specified.")
        parser.print_help()
        return

    checkpoint_filenames = args.path.split(CHECKPOINT_PATHS_DELIMITER)

    beam_search = BeamSearch.build_from_checkpoints(
        checkpoint_filenames=checkpoint_filenames,
        src_dict_filename=args.src_dict,
        dst_dict_filename=args.dst_dict,
        beam_size=args.beam_size,
        word_reward=args.word_reward,
        unk_reward=args.unk_reward,
    )
    beam_search.save_to_pytorch(output_path=args.output_file)
    if args.output_graph_file:
        with open(args.output_graph_file.path, "w") as f:
            f.write(str(beam_search.graph))


if __name__ == "__main__":
    main()
