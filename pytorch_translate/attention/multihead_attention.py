#!/usr/bin/env python3


from fairseq.modules import multihead_attention as fair_multihead
from pytorch_translate.attention import (
    BaseAttention,
    attention_utils,
    register_attention,
)


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
        src_length_mask=True
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

    def forward(self, decoder_state, source_hids, src_lengths):
        batch_size = decoder_state.shape[0]
        query = decoder_state.unsqueeze(1).transpose(0, 1)
        value = key = source_hids

        src_len_mask = None
        if src_lengths is not None and self.use_src_length_mask:
            # [batch_size, 1, seq_len]
            src_len_mask_int = attention_utils.create_src_lengths_mask(
                batch_size=batch_size, src_lengths=src_lengths
            )
            src_len_mask = src_len_mask_int != 1

        attn, attn_weights = self._fair_attn.forward(
            query, key, value, key_padding_mask=src_len_mask, need_weights=True
        )

        # attn.shape = tgt_len X bsz X embed_dim
        # attn_weights.shape = src_len X tgt_len X bsz
        return attn.squeeze(0), attn_weights.transpose(0, 2).squeeze(1)
