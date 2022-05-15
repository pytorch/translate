#!/usr/bin/env python3

import torch
from pytorch_translate.attention import (
    attention_utils,
    BaseAttention,
    register_attention,
)
from torch.autograd import Variable


@register_attention("pooling")
class PoolingAttention(BaseAttention):
    def __init__(self, decoder_hidden_state_dim, context_dim, **kwargs):
        super().__init__(decoder_hidden_state_dim, context_dim)

        self.pool_type = kwargs.get("pool_type", "mean")

    def forward(self, decoder_state, source_hids, src_lengths):
        assert self.decoder_hidden_state_dim == self.context_dim
        max_src_len = source_hids.size()[0]
        assert max_src_len == src_lengths.data.max()
        batch_size = source_hids.size()[1]

        src_mask = (
            attention_utils.create_src_lengths_mask(batch_size, src_lengths)
            .type_as(source_hids)
            .t()
            .unsqueeze(2)
        )

        if self.pool_type == "mean":
            # need to make src_lengths a 3-D tensor to normalize masked_hiddens
            denom = src_lengths.view(1, batch_size, 1).type_as(source_hids)
            masked_hiddens = source_hids * src_mask
            context = (masked_hiddens / denom).sum(dim=0)
        elif self.pool_type == "max":
            masked_hiddens = source_hids - 10e6 * (1 - src_mask)
            context = masked_hiddens.max(dim=0)[0]
        else:
            raise ValueError(f"Pooling type {self.pool_type} is not supported.")
        attn_scores = Variable(
            torch.ones(src_mask.shape[1], src_mask.shape[0]).type_as(source_hids.data),
            requires_grad=False,
        ).t()

        return context, attn_scores


@register_attention("max")
class MaxPoolingAttention(PoolingAttention):
    def __init__(self, decoder_hidden_state_dim, context_dim, **kwargs):
        super().__init__(decoder_hidden_state_dim, context_dim, pool_type="max")


@register_attention("mean")
class MeanPoolingAttention(PoolingAttention):
    def __init__(self, decoder_hidden_state_dim, context_dim, **kwargs):
        super().__init__(decoder_hidden_state_dim, context_dim, pool_type="mean")
