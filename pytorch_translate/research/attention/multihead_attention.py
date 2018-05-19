#!/usr/bin/env python3

import numpy as np
import math
import torch
import torch.nn.functional as F


def create_src_lengths_mask(batch_size, src_lengths):
    max_srclen = src_lengths.max()
    src_indices = torch.arange(
        0,
        max_srclen,
    ).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_srclen)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(
        batch_size,
        max_srclen,
    )
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()


def apply_masks(
    scores,
    batch_size,
    unseen_mask,
    src_lengths,
):
    seq_len = scores.shape[-1]

    # [1, seq_len, seq_len]
    sequence_mask = torch.ones(seq_len, seq_len).unsqueeze(0).int()
    if unseen_mask:
        # [1, seq_len, seq_len]
        sequence_mask = torch.tril(
            torch.ones(seq_len, seq_len),
            diagonal=0,
        ).unsqueeze(0).int()

    if src_lengths is not None:
        # [batch_size, 1, seq_len]
        src_lengths_mask = create_src_lengths_mask(
            batch_size=batch_size,
            src_lengths=src_lengths,
        ).unsqueeze(-2)

        # [batch_size, seq_len, seq_len]
        sequence_mask = sequence_mask & src_lengths_mask

    # [batch_size, 1, seq_len, seq_len]
    sequence_mask = sequence_mask.unsqueeze(1)

    scores = scores.masked_fill(sequence_mask == 0, -np.inf)
    return scores


def scaled_dot_prod_attn(
    query,
    key,
    value,
    unseen_mask=False,
    src_lengths=None,
):
    """
    Scaled Dot Product Attention

    Implements equation:
    Attention(Q, K, V) = softmax(QK^T/\sqrt{d_k})V

    Inputs:
      query : [batch size, nheads, sequence length, d_k]
      key : [batch size, nheads, sequence length, d_k]
      value : [batch size, nheads, sequence length, d_v]
      unseen_mask: if True, only attend to previous sequence positions
      src_lengths_mask: if True, mask padding based on src_lengths

    Outputs:
      attn: [batch size, sequence length, d_v]

    Note that in this implementation d_q = d_k = d_v = dim
    """
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(d_k)
    if unseen_mask or src_lengths is not None:
        scores = apply_masks(
            scores=scores,
            batch_size=query.shape[0],
            unseen_mask=unseen_mask,
            src_lengths=src_lengths,
        )
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn
