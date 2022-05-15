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


@register_attention("dot")
class DotAttention(BaseAttention):
    def __init__(self, decoder_hidden_state_dim, context_dim, **kwargs):
        super().__init__(decoder_hidden_state_dim, context_dim)

        self.input_proj = None
        force_projection = kwargs.get("force_projection", False)
        if force_projection or decoder_hidden_state_dim != context_dim:
            self.input_proj = Linear(decoder_hidden_state_dim, context_dim, bias=True)
        self.src_length_masking = kwargs.get("src_length_masking", True)

    def prepare_for_onnx_export_(self, **kwargs):
        self.src_length_masking = False

    def forward(self, decoder_state, source_hids, src_lengths):
        # Reshape to bsz x src_len x context_dim
        source_hids = source_hids.transpose(0, 1)
        # decoder_state: bsz x context_dim
        if self.input_proj is not None:
            decoder_state = self.input_proj(decoder_state)
        # compute attention (bsz x src_len x context_dim) * (bsz x context_dim x 1)
        attn_scores = torch.bmm(source_hids, decoder_state.unsqueeze(2)).squeeze(2)

        # Mask + softmax (bsz x src_len)
        normalized_masked_attn_scores = attention_utils.masked_softmax(
            attn_scores, src_lengths, self.src_length_masking
        )

        # Sum weighted sources
        attn_weighted_context = (
            (source_hids * normalized_masked_attn_scores.unsqueeze(2))
            .contiguous()
            .sum(1)
        )

        return attn_weighted_context, normalized_masked_attn_scores.t()
