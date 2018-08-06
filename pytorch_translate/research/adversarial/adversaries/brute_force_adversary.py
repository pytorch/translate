#!/usr/bin/env python3

import torch
import numpy as np

from . import BaseAdversary, register_adversary


@register_adversary("brute_force")
class BruteForceAdversary(BaseAdversary):
    """This adversary just flips a fix amount of words so as to maximize the
    criterion"""

    def __init__(self, args, model, task):
        super().__init__(args, model, task)
        self.encoder = model.encoder
        self.max_swaps = args.max_swaps

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--normalize-directions",
            action="store_true",
            default=False,
            help="Take the dot product between the gradient and the normalized"
            "swap directions. When this is off the adversary will favor swaps"
            "with word vectors that are far away (because the magnitude of the"
            "dot product will be higher)",
        )

    def compute_distance(self, src_embeds, vocab_embeds):
        """Compute L2 distance between each word in the vocab and each word
        in the source"""
        # We will compute the squared norm first to avoid having to compute all
        # the directions (which would have space complexity B x T x |V| x d)
        # First compute the squared norm of each word vector
        vocab_sq_norm = vocab_embeds.norm(p=2, dim=-1)**2
        src_sq_norm = src_embeds.norm(p=2, dim=-1)**2
        # Take the dot product
        dot_product = torch.einsum("bij,kj->bik", (src_embeds, vocab_embeds))
        # Reshape for broadcasting
        # 1 x 1 x |V|
        vocab_sq_norm = vocab_sq_norm.unsqueeze(0).unsqueeze(0)
        # B x T x 1
        src_sq_norm = src_sq_norm.unsqueeze(2)
        # compute squared difference
        sq_norm = vocab_sq_norm + src_sq_norm - 2 * dot_product
        # Relu + epsilon for numerical stability
        sq_norm = torch.nn.functional.relu(sq_norm) + 1e-20
        # Take the square root
        return sq_norm.sqrt()

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
        # Replacing word x_t with word w will move the input in direction (w - x_t)
        # We want to find such that grad[x_t]^T * (w - x_t) is **minimized**
        # (because we minimize the adversarial criterion).
        # 1. Take grad[x_i]^T * w_j for each position i in the source sentences
        #    and each potential replacement w_j. We are multiplying a B x T x d
        #    tensor with a |V| x d
        # new_embed_dot_grad = input_gradients.bmm(embedding_matrix.t())
        new_embed_dot_grad = torch.einsum(
            "bij,kj->bik", (input_gradients, embedding_matrix)
        )
        # 2. Now take grad[x_i]^T * x_i for each position i. We do this with
        #    einsum: multiply all dimensions together and sum along the last one
        #    (the word embedding dim)
        prev_embed_dot_grad = torch.einsum(
            "bij,bij->bi", (input_gradients, src_embeds)
        )
        # 3. Take the difference for each possible word. The resulting tensor is
        #    of shape B x T x |V|. The value at index b,t,i is the dot product
        #    between the gradient wrt. step t of batch element b and the
        #    difference between the word embedding at step t and the embedding
        #    of word i
        direction_dot_grad = new_embed_dot_grad - prev_embed_dot_grad.unsqueeze(-1)
        # 3.5 Renormalize if necessary
        if self.args.normalize_directions:
            direction_norm = self.compute_distance(src_embeds, embedding_matrix)
            direction_dot_grad /= direction_norm
        # 4. Next we find the best substitution at each step by taking the min
        score_at_each_step, best_at_each_step = direction_dot_grad.min(2)
        # 4.5. Mask the special tokens with +infinity to make sure we won't touch them
        #      them
        token_is_special = torch.lt(src_tokens, self.src_dict.nspecial)
        masked_score_at_each_step = score_at_each_step.masked_fill(
            token_is_special, np.inf
        )
        # 5. Pick the best positions (the topk smallest scores)
        _, best_positions = masked_score_at_each_step.topk(
            max_swaps, largest=False
        )
        # 6. Create adversarial examples.
        adv_tokens = src_tokens.detach()
        batch_range = torch.arange(0, batch_size).long()
        # Swap the tokens (one position at a time)
        for swap_id in range(max_swaps):
            swap_positions = best_positions[:, swap_id]
            new_tokens = best_at_each_step[batch_range, swap_positions]
            adv_tokens[batch_range, swap_positions] = new_tokens
        # Return
        return adv_tokens
