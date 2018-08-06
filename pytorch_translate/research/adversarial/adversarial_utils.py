#!/usr/bin/env python3

import re

import torch
import torch.nn.functional as F

# Delimiter for the word to word file
blank_delim = re.compile(r"[ \t]+")


def pairwise_dot_product(src_embeds, vocab_embeds, cosine=False):
    """Compute the cosine similarity between each word in the vocab and each
    word in the source

    If `cosine=True` this returns the pairwise cosine similarity"""
    # Normlize vectors for the cosine similarity
    if cosine:
        src_embeds = F.normalize(src_embeds, dim=-1, p=2)
        vocab_embeds = F.normalize(vocab_embeds, dim=-1, p=2)
    # Take the dot product
    dot_product = torch.einsum("bij,kj->bik", (src_embeds, vocab_embeds))
    return dot_product


def pairwise_distance(src_embeds, vocab_embeds, squared=False):
    """Compute the euclidean distance between each word in the vocab and each
    word in the source"""
    # We will compute the squared norm first to avoid having to compute all
    # the directions (which would have space complexity B x T x |V| x d)
    # First compute the squared norm of each word vector
    vocab_sq_norm = vocab_embeds.norm(p=2, dim=-1) ** 2
    src_sq_norm = src_embeds.norm(p=2, dim=-1) ** 2
    # Take the dot product
    dot_product = pairwise_dot_product(src_embeds, vocab_embeds)
    # Reshape for broadcasting
    # 1 x 1 x |V|
    vocab_sq_norm = vocab_sq_norm.unsqueeze(0).unsqueeze(0)
    # B x T x 1
    src_sq_norm = src_sq_norm.unsqueeze(2)
    # Compute squared difference
    sq_norm = vocab_sq_norm + src_sq_norm - 2 * dot_product
    # Either return the squared norm or return the sqrt
    if squared:
        return sq_norm
    else:
        # Relu + epsilon for numerical stability
        sq_norm = F.relu(sq_norm) + 1e-20
        # Take the square root
        return sq_norm.sqrt()


def sample_gumbel_trick(logits, temperature=1.0, num_samples=None, dim=-1):
    """Use the gumbel trick to sample from a distribution parametrized by logits

    For references on the Gumbel trick see eg.:
    - Original paper:
    stat.ucla.edu/~gpapan/pubs/confr/PapandreouYuille_PerturbAndMap_ieee-c-iccv11.pdf
    - Nice blog post:
    http://irenechen.net/blog/2017/08/17/gumbel-trick.html
    """
    # Sample from the Gumbel distribution
    uniform_noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)
    # For stable behaviour at both high and low temperature we either rescale
    # the logprob or the gumbel noise so that we never end up with huge noise
    # or logits. In any case, all the following expressions are equal up to
    # a multiplicative constant (which doesn't matter because we're only
    # interested in the softmax)
    if temperature > 1:
        # High temperature: rescale the logits
        noisy_logits = logits / temperature + gumbel_noise
    elif temperature == 1:
        # Temperature = 1: no rescaling needed
        noisy_logits = logits + gumbel_noise
    else:
        # Low temperatures: rescale the noise
        noisy_logits = logits + gumbel_noise * temperature
    # Use the Gumbel trick to sample
    if num_samples is None:
        # The behavior is different for num_samples=None and num_samples=1
        # if num_samples=None we reduce the singleton dim
        _, samples = noisy_logits.max(dim=dim)
    else:
        # I am not 100% sure that this is valid for more num_samples>1 (ie.
        # that it corresponds to sampling multiple times without replacement)
        _, samples = noisy_logits.topk(num_samples, dim=dim)

    return samples


def load_one_to_many_dict(filename):
    """Load a mapping from words to lists of words

    The expected format is `[word] [alternative_1] [alternative_2]...`
    The separating character can be either tab or space."""
    dic = {}
    with open(filename, "r") as dict_file:
        for line in dict_file:
            try:
                fields = blank_delim.split(line.strip())
                if len(fields) <= 1:
                    continue
                word = fields[0]
                alternatives = fields[1:]
                if word not in dic:
                    dic[word] = set()
                for alt in alternatives:
                    dic[word].add(alt)
            except ValueError:
                continue

    return dic


def tile(tensor, dim, repeat):
    """Repeat each element `repeat` times along dimension `dim`"""
    # We will insert a new dim in the tensor and torch.repeat it
    # First we get the repeating counts
    repeat_dims = [1] * len(tensor.size())
    repeat_dims.insert(dim + 1, repeat)
    # And the final dims
    new_dims = list(tensor.size())
    new_dims[dim] = 2 * tensor.size(dim)
    # Now unsqueeze, repeat and reshape
    return tensor.unsqueeze(dim + 1).repeat(*repeat_dims).view(*new_dims)


def detach_sample(sample):
    """Detach sample to save memory"""

    if len(sample) == 0:
        return {}

    def _detach(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.detach()
        elif isinstance(maybe_tensor, dict):
            return {key: _detach(val) for key, val in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_detach(val) for val in maybe_tensor]
        else:
            return maybe_tensor

    return _detach(sample)


def clone_sample(sample):
    """Clone sample to save memory"""

    if len(sample) == 0:
        return {}

    def _clone(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.clone()
        elif isinstance(maybe_tensor, dict):
            return {key: _clone(val) for key, val in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_clone(val) for val in maybe_tensor]
        else:
            return maybe_tensor

    return _clone(sample)
