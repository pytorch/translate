#!/usr/bin/env python3

import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F

from pytorch_translate.attention import (
    BaseAttention,
    register_attention,
    attention_utils,
)


def apply_masks(
    scores,
    batch_size,
    unseen_mask,
    src_lengths,
):
    seq_len = scores.shape[-1]

    # [1, batch_size, seq_len]
    sequence_mask = torch.ones(batch_size, seq_len).unsqueeze(0).int()

    if src_lengths is not None:
        # [batch_size, 1, seq_len]
        sequence_mask = attention_utils.create_src_lengths_mask(
            batch_size=batch_size,
            src_lengths=src_lengths,
        ).unsqueeze(-2)

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


def split_heads(
    X,
    nheads,
):
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
    return X.contiguous().view(
        list(X.shape[:-2]) + [nheads * d_head],
    )


@register_attention('multihead')
class MultiheadAttention(BaseAttention):
    """
    Multiheaded Scaled Dot Product Attention

    Implements equation:
    MultiHead(Q, K, V) = Concat(head_1,...,head_h)W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    Similarly to the above, d_k = d_v = d_model / h
    In this implementation, keys and values are both set to encoder output

    Inputs
      init:
        decoder_hidden_state_dim : dimensionality of decoder hidden state
        encoder_output_dim : dimensionality of encoder output
        kwargs :
          nheads : integer # of attention heads
          unseen_mask: if True, only attend to previous sequence positions
          src_lengths_mask: if True, mask padding based on src_lengths

      forward:
        decoder_state : [batch size, d_model]
        source_hids : [sequence length, batch size, d_model]
        src_lengths : [batch size]

      forward:
        query : [batch size, sequence length, d_model]
        key: [batch size, sequence length, d_model]
        value: [batch size, sequence length, d_model]

    Output
      result : [batch_size, sequence length, d_model]
    """
    def __init__(self, decoder_hidden_state_dim, encoder_output_dim, **kwargs):
        super().__init__(decoder_hidden_state_dim, encoder_output_dim)
        assert decoder_hidden_state_dim == encoder_output_dim
        d_model = decoder_hidden_state_dim  # for brevity
        self.nheads = kwargs.get("nheads", 1)
        assert d_model % self.nheads == 0
        self.d_head = d_model // self.nheads
        self.Q_fc = nn.Linear(d_model, d_model, bias=False)
        self.K_fc = nn.Linear(d_model, d_model, bias=False)
        self.V_fc = nn.Linear(d_model, d_model, bias=False)
        self.output_fc = nn.Linear(d_model, d_model, bias=False)
        self.attn = None

        self.unseen_mask = kwargs.get("unseen_mask", False)
        self.src_length_mask = kwargs.get("src_length_mask", True)

    def forward(self, decoder_state, source_hids, src_lengths):
        if self.unseen_mask:
            raise NotImplementedError(
                "Unseen mask not supported with sequential decoding"
            )

        query = decoder_state.unsqueeze(1)
        key = source_hids.transpose(0, 1)
        value = source_hids.transpose(0, 1)

        # 1. Fully-connected layer on q, k, v then
        # 2. Split heads on q, k, v
        # (batch_size, seq_len, d_model) -->
        # (batch_size, nheads, seq_len, d_head)
        query = split_heads(self.Q_fc(query), self.nheads)
        key = split_heads(self.K_fc(key), self.nheads)
        value = split_heads(self.V_fc(value), self.nheads)

        # 4. Scaled dot product attention
        # (batch_size, nheads, seq_len, d_head)
        if not self.src_length_mask:
            src_lengths = None
        x, self.attn = scaled_dot_prod_attn(
            query=query,
            key=key,
            value=value,
            unseen_mask=self.unseen_mask,
            src_lengths=src_lengths,
        )

        # 5. Combine heads
        x = combine_heads(x)

        # 6. Average attention weights for all heads
        attn_aggregated = self.attn.sum(dim=1) / self.nheads

        # 7. Fully-connected layer for output
        return self.output_fc(x).squeeze(1), attn_aggregated.squeeze(1)
