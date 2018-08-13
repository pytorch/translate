#!/usr/bin/env python3

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def create_src_lengths_mask(batch_size, src_lengths):
    max_srclen = src_lengths.max()
    src_indices = torch.arange(0, max_srclen).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_srclen)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_srclen)
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()


def apply_masks(scores, batch_size, unseen_mask, src_lengths):
    seq_len = scores.shape[-1]

    # [1, seq_len, seq_len]
    sequence_mask = torch.ones(seq_len, seq_len).unsqueeze(0).int()
    if unseen_mask:
        # [1, seq_len, seq_len]
        sequence_mask = (
            torch.tril(torch.ones(seq_len, seq_len), diagonal=0).unsqueeze(0).int()
        )

    if src_lengths is not None:
        # [batch_size, 1, seq_len]
        src_lengths_mask = create_src_lengths_mask(
            batch_size=batch_size, src_lengths=src_lengths
        ).unsqueeze(-2)

        # [batch_size, seq_len, seq_len]
        sequence_mask = sequence_mask & src_lengths_mask

    # [batch_size, 1, seq_len, seq_len]
    sequence_mask = sequence_mask.unsqueeze(1)

    scores = scores.masked_fill(sequence_mask == 0, -np.inf)
    return scores


def scaled_dot_prod_attn(query, key, value, unseen_mask=False, src_lengths=None):
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


def split_heads(X, nheads):
    """
    Split heads:
    1) Split (reshape) last dimension (size d_model) into nheads, d_head
    2) Transpose X from (batch size, sequence length, nheads, d_head) to
        (batch size, nheads, sequence length, d_head)

    Inputs:
      X : [batch size, sequence length, nheads * d_head]
      nheads : integer
    Outputs:
      [batch size,  nheads, sequence length, d_head]

    """
    last_dim = X.shape[-1]
    assert last_dim % nheads == 0
    X_last_dim_split = X.view(list(X.shape[:-1]) + [nheads, last_dim // nheads])
    return X_last_dim_split.transpose(1, 2)


def combine_heads(X):
    """
    Combine heads (the inverse of split heads):
    1) Transpose X from (batch size, nheads, sequence length, d_head) to
        (batch size, sequence length, nheads, d_head)
    2) Combine (reshape) last 2 dimensions (nheads, d_head) into 1 (d_model)

    Inputs:
      X : [batch size * nheads, sequence length, d_head]
      nheads : integer
      d_head : integer

    Outputs:
      [batch_size, seq_len, d_model]

    """
    X = X.transpose(1, 2)
    nheads, d_head = X.shape[-2:]
    return X.contiguous().view(list(X.shape[:-2]) + [nheads * d_head])


class MultiheadAttention(nn.Module):
    """
    Multiheaded Scaled Dot Product Attention

    Implements equation:
    MultiHead(Q, K, V) = Concat(head_1,...,head_h)W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    Similarly to the above, d_k = d_v = d_model / h

    Inputs
      init:
        nheads : integer # of attention heads
        d_model : model dimensionality
        d_head : dimensionality of a single head

      forward:
        query : [batch size, sequence length, d_model]
        key: [batch size, sequence length, d_model]
        value: [batch size, sequence length, d_model]
      unseen_mask: if True, only attend to previous sequence positions
      src_lengths_mask: if True, mask padding based on src_lengths

    Output
      result : [batch_size, sequence length, d_model]
    """

    def __init__(self, nheads, d_model):
        "Take in model size and number of heads."
        super(MultiheadAttention, self).__init__()
        assert d_model % nheads == 0
        self.d_head = d_model // nheads
        self.nheads = nheads
        self.Q_fc = nn.Linear(d_model, d_model, bias=False)
        self.K_fc = nn.Linear(d_model, d_model, bias=False)
        self.V_fc = nn.Linear(d_model, d_model, bias=False)
        self.output_fc = nn.Linear(d_model, d_model, bias=False)
        self.attn = None

    def forward(self, query, key, value, unseen_mask=False, src_lengths=None):
        # 1. Fully-connected layer on q, k, v then
        # 2. Split heads on q, k, v
        # (batch_size, seq_len, d_model) -->
        # (batch_size, nheads, seq_len, d_head)
        query = split_heads(self.Q_fc(query), self.nheads)
        key = split_heads(self.K_fc(key), self.nheads)
        value = split_heads(self.V_fc(value), self.nheads)

        # 4. Scaled dot product attention
        # (batch_size, nheads, seq_len, d_head)
        x, self.attn = scaled_dot_prod_attn(
            query=query,
            key=key,
            value=value,
            unseen_mask=unseen_mask,
            src_lengths=src_lengths,
        )

        # 5. Combine heads
        x = combine_heads(x)

        # 6. Fully-connected layer for output
        return self.output_fc(x)
