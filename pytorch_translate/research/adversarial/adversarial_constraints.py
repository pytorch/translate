#!/usr/bin/env python3

import torch
import numpy as np


from .adversarial_utils import (
    pairwise_dot_product,
    pairwise_distance,
    load_one_to_many_dict,
)


class AdversarialConstraints(object):
    """This is an abstraction to represent the constraints imposed on the
    adversary"""

    def __init__(self, args, task):
        self.args = args
        self.src_dict = task.src_dict
        self.tgt_dict = task.tgt_dict
        self.nearest_neighbors = self.args.nearest_neighbors
        self.cosine_nn = self.args.cosine_nn
        self.alternatives_file = self.args.alternatives_file
        # Load the explicit alternatives
        if self.alternatives_file:
            self.load_alternatives()
        # Forbidden words
        self.forbidden_tokens = torch.LongTensor([
            self.src_dict.index(word) for word in self.args.forbidden_words
        ])
        if torch.cuda.is_available():
            self.forbidden_tokens = self.forbidden_tokens.cuda()

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--nearest-neighbors",
            "--nearest-neighbours",  # for our british friends #inclusivity
            metavar="K",
            type=int,
            default=0,
            help="Only allow replacing a word with one of its nearest neighbors."
            " This is only applied if K>0.",
        )
        parser.add_argument(
            "--cosine-nn",
            action="store_true",
            default=False,
            help="Use the cosine distance instead of l2 for the cosine similarity",
        )
        parser.add_argument(
            "--alternatives-file",
            metavar="FILE",
            type=str,
            default="",
            help="file specifying word replacement alternatives in the format: "
            "word replacement_1 replacement_2 ... \nThe file can be space or "
            "tab separated.",
        )
        parser.add_argument(
            "--forbidden-words",
            nargs="*",
            type=str,
            default=[],
            help="List of words that the model absolutely cannot produce.",
        )

    def apply(self, scores, src_tokens, src_embeds, vocab_embeds, **kwargs):
        """This assigns a score of -inf to replacements that are not allowed
        under the constraints

        `scores` is a B x T x |V| tensor such that `scores[b, t, w]` is high (
        resp. low) if replacing the word at position `t` in batch element `b`
        by `w` is good (resp. bad) for the adversary. Having a score of -inf
        means that the choice will never be made.

        **kwargs is used to pass existing variable to prevent doing expensive
        operations again (typically if the pairwise distance has been computed
        already no need to do it again for the knn)"""

        # Replacing a word with itself is NEVER a good idea
        scores.scatter_(dim=2, index=src_tokens.unsqueeze(-1), value=-np.inf)
        # We don't want to touch special tokens either
        token_is_special = torch.lt(src_tokens, self.src_dict.nspecial)
        # Except unk
        token_is_unk = torch.eq(src_tokens, self.src_dict.unk())
        scores.masked_fill_((token_is_special ^ token_is_unk).unsqueeze(-1), -np.inf)
        # Nor replace an existing word with a special token (except <unk>)
        special_tokens = torch.tensor([
            token_id for token_id in range(self.src_dict.nspecial)
            if token_id != self.src_dict.unk()
        ], device=scores.device).long()
        scores.index_fill_(dim=2, index=special_tokens, value=-np.inf)

        # Forbid some words specifically
        if len(self.forbidden_tokens) > 0:
            scores.index_fill_(dim=2, index=self.forbidden_tokens, value=-np.inf)
        # Now we address the explicit constraints of which words are authorized
        # This will contain a list of tensors of shape B x T x _
        # each corresponding to allowed indices
        allowed = []
        # kNNs
        if self.nearest_neighbors > 1:
            allowed.append(self.compute_knn(src_embeds, vocab_embeds))

        # Explicit alternatives
        if self.alternatives_file:
            # We pick alt_tokens[b, t, i] = alts_tensor[src_tokens[b, t], i]
            # We do it on the flattened tensor:
            # alt_tokens[b * t, i] = alts_tensor[src_tokens[b * t], i]
            alt_tokens = self.alts_tensor.index_select(dim=0, index=src_tokens.view(-1))
            # Reshape into a B x T x max_swaps tensor
            bsz, srclen = src_tokens.size()
            alt_tokens = alt_tokens.view(bsz, srclen, -1)
            # add to the allowed tokens
            allowed.append(alt_tokens)
        # If nothing is allowed then everything is allowed so we do NOTHING.
        # Otherwise:
        if len(allowed) > 0:
            # Concatenate all allowed tokens at each position b,t
            allowed_ids = torch.cat(allowed, dim=2)
            # Now create a mask of shape B x T x |V|
            not_allowed = torch.ones_like(scores, dtype=torch.uint8)
            # Make sure the allowed words have a mask of 0
            # (duplicates are handled here)
            not_allowed.scatter_(dim=2, index=allowed_ids, value=0)
            # Now mask the scores of forbidden words with -infinity
            scores.masked_fill_(not_allowed, -np.inf)

    def compute_knn(self, src_embeds, vocab_embeds, **kwargs):
        """Compute k nearest neighbors of each embedding in the source"""
        # Compute or retrieve distance
        if self.cosine_nn:
            # Cosine similarity
            if "cosine_distance" in kwargs:
                distances = kwargs["cosine_distance"]
            else:
                distances = -pairwise_dot_product(src_embeds, vocab_embeds, cosine=True)
        else:
            # L2 distance
            if "l2_distance" in kwargs:
                distances = kwargs["l2_distance"]
            else:
                distances = pairwise_distance(src_embeds, vocab_embeds)
        # Get the knn
        _, knn = distances.topk(
            # Add +1 to account for the word itself
            self.args.nearest_neighbors + 1,
            dim=2,
            largest=False,
        )
        # Return the kNN minus the word itself
        return knn[:, :, 1:]

    def load_alternatives(self):
        """Load alternative words from a file and store the in a tensor"""
        alts_dict = load_one_to_many_dict(self.alternatives_file)
        # Numberize and remove duplicate IDs (typically there will be a lot of
        # UNKs)
        alts_ids_dict = {}
        for word, alts in alts_dict.items():
            # Retrieve IDs
            word_id = self.src_dict.index(word)
            # Ignore unknow words
            if word_id == self.src_dict.unk_index:
                continue

            alts_ids = {self.src_dict.index(alt) for alt in alts}
            alts_ids_dict[word_id] = list(alts_ids)
        # It's easier in pytorch to have all the words with the same number
        # of alternatives, so we pad by copying. It is OK for us to have
        # duplicates.
        max_alts = max(len(alts) for alts in alts_ids_dict.values())
        for word in alts_ids_dict.keys():
            while len(alts_ids_dict[word]) < max_alts:
                alts_ids_dict[word].append(alts_ids_dict[word][-1])
        # Now we create the tensor. By default the `max_alts` alternatives for
        # word w are w itself.
        self.alts_tensor = (
            torch.arange(len(self.src_dict)).unsqueeze(-1).repeat(1, max_alts)
        )
        # Now we add the actual alteratives
        for word_id, alts_ids in alts_ids_dict.items():
            # Assign value
            self.alts_tensor[word_id] = torch.LongTensor(alts_ids)

        if torch.cuda.is_available():
            self.alts_tensor = self.alts_tensor.cuda()
