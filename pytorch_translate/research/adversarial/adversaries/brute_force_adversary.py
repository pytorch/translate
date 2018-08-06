#!/usr/bin/env python3

import torch
import torch.nn.functional as F

from . import BaseAdversary, register_adversary
from ..adversarial_utils import pairwise_distance


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
        # so that grad[x_t]^T * (x_t - w) is **maximized** (because we minimize
        # the adversarial criterion).
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
        #    difference between the embedding of word i and the word embedding
        #    at step t
        direction_dot_grad = prev_embed_dot_grad.unsqueeze(-1) - new_embed_dot_grad
        # 3.5 Renormalize if necessary
        if self.args.normalize_directions:
            # Compute the direction norm (= distance word/substitution)
            direction_norm = pairwise_distance(src_embeds, embedding_matrix)
            # Renormalize
            direction_dot_grad /= direction_norm
        # 4. Apply constraints
        self.constraints.apply(
            direction_dot_grad,
            src_tokens,
            src_embeds,
            embedding_matrix
        )
        # 4. Next we find the best substitution at each step by taking the max
        score_at_each_step, best_at_each_step = direction_dot_grad.max(2)
        # 5. Pick the best positions (the topk highest scores)
        _, best_positions = score_at_each_step.topk(
            max_swaps
        )
        # 6. Create adversarial examples.
        adv_tokens = src_tokens.detach()
        # Assign new values
        # adv_tokens[b, best_positions[t]]=best_at_each_step[b, best_positions[t]]
        adv_tokens.scatter_(
            dim=1,
            index=best_positions,
            src=best_at_each_step.gather(dim=1, index=best_positions)
        )
        # Return
        return adv_tokens
