import random
import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

WORD_DROPOUT_SMOOTHING_ALPHA_DEFAULT = 1


def add_args(parser):
    parser.add_argument(
        '--word_dropout_freq_threshold',
        default=None,
        type=int,
        metavar='N',
        help=('Frequency threshold under which word dropout is performed.')
    )

    parser.add_argument(  # set alpha to 1 for no smoothing
        '--word_dropout_smoothing_alpha',
        type=int,
        metavar='N',
        help=('Smooth word dropout using alpha/(freq+alpha).')
    )


def set_arg_defaults(args):
    if hasattr(args, 'word_dropout_params'):
        # We've already created the word dropout params from the bottom-level
        # args (word_dropout_freq, word_dropout_smoothing_alpha)
        return args.word_dropout_params
    args.word_dropout_params = None
    word_dropout_freq_threshold = getattr(
        args,
        'word_dropout_freq_threshold',
    )
    if word_dropout_freq_threshold is not None:
        word_dropout_smoothing_alpha = getattr(
            args,
            'word_dropout_smoothing_alpha',
            WORD_DROPOUT_SMOOTHING_ALPHA_DEFAULT,
        )
        args.word_dropout_params = {
            'word_dropout_freq_threshold': word_dropout_freq_threshold,
            'word_dropout_smoothing_alpha': word_dropout_smoothing_alpha,
        }

    # For less redundant logging when we print out the args Namespace,
    # delete the bottom-level args, since we'll just be dealing with
    # args.word_dropout_params from now on
    if hasattr(args, 'word_dropout_freq_threshold'):
        delattr(args, 'word_dropout_freq_threshold')
    if hasattr(args, 'word_dropout_smoothing_alpha'):
        delattr(args, 'word_dropout_smoothing_alpha')


class WordDropout(nn.Module):
    def __init__(self, src_dict, word_dropout_params):
        super().__init__()
        self.src_dict = src_dict
        self.word_dropout_params = word_dropout_params

    def forward(self, source_seq):
        for i in range(len(source_seq)):  # source_seq is a list of token ids
            for j in range(len(source_seq[i])):
                source_seq[i][j] = \
                    self.apply_probabilistic_unking(source_seq[i][j])
        return source_seq

    def apply_probabilistic_unking(self, token_id):
        word_dropout_freq_threshold = \
            self.word_dropout_params['word_dropout_freq_threshold']
        alpha = self.word_dropout_params['word_dropout_smoothing_alpha']
        if token_id >= self.src_dict.nspecial:
            freq = self.src_dict.count[token_id]
            if freq <= word_dropout_freq_threshold and \
              random.random() < float(alpha) / (freq + alpha):
                return self.src_dict.unk_index
        return token_id
