#!/usr/bin/env python3

import torch
import numpy as np

from . import BaseAdversary, register_adversary


@register_adversary("random_swap")
class RandomSwapAdversary(BaseAdversary):
    """This adversary flips words at random"""

    def __init__(self, args, model, task):
        super().__init__(args, model, task)
        self.encoder = model.encoder
        self.max_swaps = args.max_swaps
        self.temperature = args.temperature
        assert (
            self.temperature >= 0
        ), "The temperature for sampling should be a positive number"

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--by-gradient-norm",
            default=False,
            action="store_true",
            help="Sample words to swap according to the norm of their gradient.",
        )
        parser.add_argument(
            "--temperature",
            default=1.0,
            type=float,
            help="Regulate the temperature of sampling by gradient norm "
            "(high temperature -> uniform sampling/low temperature -> argmax).",
        )

    def forward(self, sample, input_gradients):
        src_tokens = sample["net_input"]["src_tokens"]
        batch_size, src_length = src_tokens.size()
        # Careful that the number of swaps is not greater than the size of the sentences
        max_swaps = min(self.max_swaps, src_length)
        # Uniform sampling
        if self.args.by_gradient_norm:
            # Compute the gradient l2 norm
            grad_norm = input_gradients.norm(p=2, dim=-1)
            # Take the log (and apply temperature)
            log_prob = torch.log(grad_norm)
        else:
            # The (non-normalized) log probability of each position will be 0
            log_prob = torch.zeros_like(src_tokens).float()
            # BUT we need to mask the special tokens with -infinity
            token_is_special = torch.lt(src_tokens, self.src_dict.nspecial)
            # apply mask
            log_prob.masked_fill(token_is_special, -np.inf)
        # Sample from the Gumbel distribution
        uniform_noise = torch.rand_like(log_prob)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)
        # For stable behaviour at both high and low temperature we either rescale
        # the logprob or the gumbel noise
        if self.temperature > 1:
            noisy_log_prob = log_prob / self.temperature + gumbel_noise
        else:
            noisy_log_prob = log_prob + gumbel_noise * self.temperature
        # Use the Gumbel trick to sample positions (not sure if valid for k>1)
        _, random_positions = noisy_log_prob.topk(max_swaps)
        # Now for each position we sample a random replacement word
        random_words = torch.randint(
            low=self.src_dict.nspecial,
            high=len(self.src_dict),
            size=(batch_size, max_swaps),
            device=src_tokens.device,
        ).long()
        # Replace words at position to get adversarial sample
        adv_tokens = src_tokens.detach()
        batch_range = torch.arange(0, batch_size)
        for swap_id in range(max_swaps):
            swap_position = random_positions[:, swap_id]
            adv_tokens[batch_range, swap_position] = random_words[:, swap_id]

        return adv_tokens
