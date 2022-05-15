#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_translate.attention import (
    attention_utils,
    BaseAttention,
    register_attention,
)
from pytorch_translate.common_layers import Linear


@register_attention("mlp")
class MLPAttention(BaseAttention):
    """The original attention from Badhanau et al. (2014)
    https://arxiv.org/abs/1409.0473 based on a Multi-Layer Perceptron.

    The attention score between position i in the encoder and position j in the
    decoder is:
    alpha_ij = V_a * tanh(W_ae * enc_i + W_ad * dec_j + b_a)
    """

    def __init__(self, decoder_hidden_state_dim, context_dim, **kwargs):
        super().__init__(decoder_hidden_state_dim, context_dim)

        self.context_dim = context_dim
        self.attention_dim = kwargs.get("attention_dim", context_dim)
        # W_ae and b_a
        self.encoder_proj = Linear(context_dim, self.attention_dim, bias=True)
        # W_ad
        self.decoder_proj = Linear(
            decoder_hidden_state_dim, self.attention_dim, bias=False
        )
        # V_a
        self.to_scores = Linear(self.attention_dim, 1, bias=False)
        self.src_length_masking = kwargs.get("src_length_masking", True)

    def prepare_for_onnx_export_(self, **kwargs):
        self.src_length_masking = False

    def forward(self, decoder_state, source_hids, src_lengths):
        """The expected input dimensions are:

        decoder_state: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        src_lengths: bsz
        """
        src_len, bsz, _ = source_hids.size()
        # (src_len*bsz) x context_dim (to feed through linear)
        flat_source_hids = source_hids.view(-1, self.context_dim)
        # (src_len*bsz) x attention_dim
        encoder_component = self.encoder_proj(flat_source_hids)
        # src_len x bsz x attention_dim
        encoder_component = encoder_component.view(src_len, bsz, self.attention_dim)
        # 1 x bsz x attention_dim
        decoder_component = self.decoder_proj(decoder_state).unsqueeze(0)
        # Sum with broadcasting and apply the non linearity
        # src_len x bsz x attention_dim
        hidden_att = F.tanh(
            (decoder_component + encoder_component).view(-1, self.attention_dim)
        )
        # Project onto the reals to get attentions scores (bsz x src_len)
        attn_scores = self.to_scores(hidden_att).view(src_len, bsz).t()

        # Mask + softmax (src_len x bsz)
        normalized_masked_attn_scores = attention_utils.masked_softmax(
            attn_scores, src_lengths, self.src_length_masking
        ).t()

        # Sum weighted sources (bsz x context_dim)
        attn_weighted_context = (
            source_hids * normalized_masked_attn_scores.unsqueeze(2)
        ).sum(0)

        return attn_weighted_context, normalized_masked_attn_scores
