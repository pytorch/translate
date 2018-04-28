#!/usr/bin/env python3

import argparse
import numpy as np
import torch

from fbtranslate import rnn  # noqa
from fbtranslate.ensemble_export import BeamSearch


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Export PyTorch-trained FBTranslate models to caffe2'
        ),
    )
    parser.add_argument(
        '--checkpoint',
        action='append',
        nargs='+',
        help='PyTorch checkpoint file (at least one required)',
    )
    parser.add_argument(
        '--output_file',
        default='',
        help='File name to which to save beam search network',
    )
    parser.add_argument(
        '--src_dict',
        required=True,
        help='File encoding PyTorch dictionary for source language',
    )
    parser.add_argument(
        '--dst_dict',
        required=True,
        help='File encoding PyTorch dictionary for source language',
    )
    parser.add_argument(
        '--beam_size',
        type=int,
        default=6,
        help='Number of top candidates returned by each decoder step',
    )
    parser.add_argument(
        '--word_penalty',
        type=float,
        default=0.0,
        help='Value to add for each word (besides EOS)',
    )
    parser.add_argument(
        '--unk_penalty',
        type=float,
        default=0.0,
        help='Value to add for each word UNK token',
    )

    args = parser.parse_args()

    if args.output_file == '':
        print('No action taken. Need output_file to be specified.')
        parser.print_help()
        return

    checkpoint_filenames = [arg[0] for arg in args.checkpoint]

    beam_search = BeamSearch.build_from_checkpoints(
        checkpoint_filenames=checkpoint_filenames,
        src_dict_filename=args.src_dict,
        dst_dict_filename=args.dst_dict,
        beam_size=args.beam_size,
        word_penalty=args.word_penalty,
        unk_penalty=args.unk_penalty,
    )

    src_dict = encoder_ensemble.models[0].src_dict
    token_list = [src_dict.unk()] * 4 + [src_dict.eos()]
    src_tokens = torch.LongTensor(
        np.array(token_list, dtype='int64').reshape(-1, 1),
    )
    src_lengths = torch.IntTensor(
        np.array([len(token_list)], dtype='int32'),
    )

    decoder_step_ensemble.save_to_db(
        args.decoder_output_file,
    )


if __name__ == '__main__':
    main()
