#!/usr/bin/env python3

import torch.nn as nn


class BaseAdversary(nn.Module):
    """Base class for adversarial example generators"""

    def __init__(self, args, model, task):
        super().__init__()
        self.task = task
        self.src_dict = task.src_dict
        self.dst_dict = task.tgt_dict
        self.args = args

    def forward(self, sample, input_gradients):
        """Takes in a sample and the gradients of some adversarial criterion
        wrt. the input tokens and returns new input tokens maximizing this criterion.
        """
        raise NotImplementedError()

    @staticmethod
    def add_args(parser):
        """Add adversary-specific arguments to the parser."""
        parser.add_argument(
            "--max-swaps",
            default=1,
            type=int,
            help="Maximum amount of words the adversary is allowed to swap.",
        )
