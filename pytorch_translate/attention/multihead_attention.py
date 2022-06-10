#!/usr/bin/env python3

from typing import Optional

from fairseq.modules import multihead_attention as fair_multihead
from pytorch_translate.attention import (
    attention_utils,
    BaseAttention,
    register_attention,
)
from torch import Tensor


@register_attention("multihead")
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
        context_dim : dimensionality of encoder output
        kwargs :
          nheads : integer # of attention heads
          unseen_mask: if True, only attend to previous sequence positions
          src_lengths_mask: if True, mask padding based on src_lengths

      forward:
        decoder_state : [batch size, d_model]
        source_hids : [sequence length, batch size, d_model]
        src_lengths : [batch size]

      forward:
        query : [sequence length, batch size, d_model]
        key: [sequence length, batch size, d_model]
        value: [sequence length, batch size, d_model]

    Output
      result : [batch_size,  d_model]
    """

    def __init__(
        self,
        decoder_hidden_state_dim,
        context_dim,
        *,
        nheads=1,
        unseen_mask=False,
        src_length_mask=True,
    ):
        super().__init__(decoder_hidden_state_dim, context_dim)
        assert decoder_hidden_state_dim == context_dim
        d_model = decoder_hidden_state_dim  # for brevity
        assert d_model % nheads == 0

        if unseen_mask:
            raise NotImplementedError(
                "Unseen mask not supported with sequential decoding"
            )
        self._fair_attn = fair_multihead.MultiheadAttention(d_model, nheads)
        self.use_src_length_mask = src_length_mask

    def forward(
        self,
        decoder_state,
        source_hids,
        src_lengths,
        squeeze: bool = True,
        max_src_len: Optional[int] = None,
    ):
        """
        Computes MultiheadAttention with respect to either a vector
        or a tensor

        Inputs:
            decoder_state: (bsz x decoder_hidden_state_dim) or
                (bsz x T x decoder_hidden_state_dim)
            source_hids: srclen x bsz x context_dim
            src_lengths: bsz x 1, actual sequence lengths
            squeeze: Whether or not to squeeze on the time dimension.
                Even if decoder_state.dim() is 2 dimensional an
                explicit time step dimension will be unsqueezed.
            max_src_len: Optionally override the max_src_len otherwise
                inferred from src_lengths. Useful during beam search when we
                might have already finalized the longest src_sequence
        Outputs:
          [batch_size, max_src_len] if decoder_state.dim() == 2 & squeeze
            or
          [batch_size, 1, max_src_len] if decoder_state.dim() == 2 & !squeeze
            or
          [batch_size, T, max_src_len] if decoder_state.dim() == 3 & !squeeze
            or
          [batch_size, T, max_src_len] if decoder_state.dim() == 3 & squeeze & T != 1
            or
          [batch_size, max_src_len] if decoder_state.dim() == 3 & squeeze & T == 1
        """
        batch_size = decoder_state.shape[0]
        if decoder_state.dim() == 3:
            query = decoder_state
        elif decoder_state.dim() == 2:
            query = decoder_state.unsqueeze(1)
        else:
            raise ValueError("decoder state must be either 2 or 3 dimensional")
        query = query.transpose(0, 1)
        value = key = source_hids

        src_len_mask: Optional[Tensor] = None
        if src_lengths is not None and self.use_src_length_mask:
            # [batch_size, 1, seq_len]
            src_len_mask_int = attention_utils.create_src_lengths_mask(
                batch_size=batch_size, src_lengths=src_lengths, max_src_len=max_src_len
            )
            src_len_mask = src_len_mask_int != 1
        attn, attn_weights = self._fair_attn.forward(
            query, key, value, key_padding_mask=src_len_mask, need_weights=True
        )
        # attn.shape = T X bsz X embed_dim
        # attn_weights.shape = bsz X T X src_len
        if attn_weights is not None:
            attn_weights = attn_weights.transpose(0, 2)
        # attn_weights.shape = src_len X T X bsz

        if squeeze:
            attn = attn.squeeze(0)
            # attn.shape = squeeze(T) X bsz X embed_dim
            if attn_weights is not None:
                attn_weights = attn_weights.squeeze(1)
            # attn_weights.shape = src_len X squeeze(T) X bsz
            return attn, attn_weights
        return attn, attn_weights
