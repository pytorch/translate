#!/usr/bin/env python3

import torch

from . import BaseAdversary, register_adversary
from ..adversarial_utils import sample_gumbel_trick


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
        # Embed input tokens (we could reuse the embeddings here if it doesn't
        # make the interface too complicated)
        src_embeds = self.encoder.embed_tokens(src_tokens)
        # Get the embeddings for the whole vocabulary |V| x d
        embedding_matrix = self.encoder.embed_tokens.weight
        # 1. Get the logits for sampling
        if self.args.by_gradient_norm:
            # Compute the gradient l2 norm
            grad_norm = input_gradients.norm(p=2, dim=-1) + 1e-12
            # Take the log (and apply temperature)
            logits = torch.log(grad_norm)
        else:
            # The (non-normalized) log probability of each position will be 0
            logits = torch.zeros_like(src_tokens).float()
        # 2. Expand logits to shape B x T x |V|to apply constraints
        logits = logits.unsqueeze(-1).expand(-1, -1, len(self.src_dict)).clone()
        # 2.5. Apply constraints
        self.constraints.apply(logits, src_tokens, src_embeds, embedding_matrix)
        # 3. Get the logits for the positions
        logits_pos, _ = logits.max(dim=2)
        # 3.5 Sample positions with the Gumbel trick
        random_positions = sample_gumbel_trick(
            logits_pos, temperature=self.temperature, num_samples=max_swaps, dim=1
        )
        # 4. Now for each position we sample a random replacement word
        # We still need to do gumbel sampling to account for the -inf log probs
        # corresponding to the forbidden tokens.
        # word_logits[b, k, w] = logits[b, random_positions[b, k], w]
        word_logits = logits.gather(
            dim=1,
            # Expand the position indices (can we do more efficient than this??)
            index=random_positions.unsqueeze(-1).expand(-1, -1, len(self.src_dict))
        )
        # Sample words with the Gumbel trick
        random_words = sample_gumbel_trick(word_logits, dim=2)
        # 5. Create adversarial examples.
        adv_tokens = src_tokens.clone()
        # Assign new values
        # adv_tokens[b, random_positions[b, k]]=random_words[b, random_positions[b, k]]
        adv_tokens.scatter_(
            dim=1,
            index=random_positions,
            src=random_words,
        )
        # Return
        return adv_tokens
