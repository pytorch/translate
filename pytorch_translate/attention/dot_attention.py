#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_translate.attention import (
    BaseAttention,
    register_attention,
    attention_utils,
)
from pytorch_translate.common_layers import Linear


@register_attention('dot')
class DotAttention(BaseAttention):

    def __init__(self, decoder_hidden_state_dim, encoder_output_dim, **kwargs):
        super().__init__(decoder_hidden_state_dim, encoder_output_dim)

        self.input_proj = None
        force_projection = kwargs.get("force_projection", False)
        if force_projection or decoder_hidden_state_dim != encoder_output_dim:
            self.input_proj = Linear(
                decoder_hidden_state_dim, encoder_output_dim, bias=True
            )
        self.src_length_masking = kwargs.get("src_length_masking", True)

    def forward(self, decoder_state, source_hids, src_lengths):
        # decoder_state: bsz x encoder_output_dim
        if self.input_proj is not None:
            decoder_state = self.input_proj(decoder_state)
        # compute attention
        attn_scores = (source_hids * decoder_state.unsqueeze(0)).sum(dim=2).t()

        if self.src_length_masking:
            max_src_len = source_hids.size()[0]
            assert max_src_len == src_lengths.data.max()
            batch_size = source_hids.size()[1]
            src_mask = attention_utils.create_src_lengths_mask(
                batch_size,
                src_lengths,
            )
            masked_attn_scores = attn_scores.masked_fill(src_mask == 0, -np.inf)
            # Since input of varying lengths, need to make sure the attn_scores
            # for each sentence sum up to one
            attn_scores = F.softmax(masked_attn_scores, dim=-1)  # bsz x srclen
            score_denom = torch.sum(attn_scores, dim=1).unsqueeze(dim=1).expand(
                batch_size, max_src_len
            )
            normalized_masked_attn_scores = torch.div(attn_scores, score_denom).t()
        else:
            normalized_masked_attn_scores = F.softmax(attn_scores, dim=-1).t()

        # sum weighted sources
        attn_weighted_context = (
            source_hids * normalized_masked_attn_scores.unsqueeze(2)
        ).sum(
            dim=0
        )

        return attn_weighted_context, normalized_masked_attn_scores
