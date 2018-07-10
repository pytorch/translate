#!/usr/bin/env python3

import torch
from pytorch_translate.attention import BaseAttention, register_attention
from pytorch_translate.utils import maybe_cuda


@register_attention("no")
class NoAttention(BaseAttention):
    def __init__(self, decoder_hidden_state_dim, context_dim, **kwargs):
        super().__init__(decoder_hidden_state_dim, 0)

    def forward(self, decoder_state, source_hids, src_lengths):
        return None, maybe_cuda(torch.zeros(1, src_lengths.shape[0]))
