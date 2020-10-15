#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from torch import nn


@with_incremental_state
class AttentionAbstract(nn.Module):
    """Abstract class for attention modules"""

    def __init__(self):
        super().__init__()
        self.incremental_clone_ids = set("")

    def forward(
        self,
        query,
        key,
        value,
        mask_future_timesteps=False,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
    ):
        raise NotImplementedError()

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        for incremental_clone_id in self.incremental_clone_ids:
            input_buffer = self._get_input_buffer(
                incremental_state, incremental_clone_id=incremental_clone_id
            )
            if input_buffer is not None:
                for k in input_buffer.keys():
                    if torch.is_tensor(input_buffer[k]) and input_buffer[k].size(1) > 1:
                        input_buffer[k] = input_buffer[k].index_select(1, new_order)
                self._set_input_buffer(
                    incremental_state,
                    input_buffer,
                    incremental_clone_id=incremental_clone_id,
                )

    def _get_input_buffer(self, incremental_state, incremental_clone_id: str = ""):
        return (
            utils.get_incremental_state(
                self, incremental_state, "attn_state" + incremental_clone_id
            )
            or {}
        )

    def _set_input_buffer(
        self, incremental_state, buffer, incremental_clone_id: str = ""
    ):
        self.incremental_clone_ids.add(incremental_clone_id)
        utils.set_incremental_state(
            self, incremental_state, "attn_state" + incremental_clone_id, buffer
        )


class AverageAttention(AttentionAbstract):
    """Average attention.
    See "Accelerating Neural Transformer via an Average Attention Network"
    for more details.
    """

    def __init__(self, embed_dim, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

    def forward(
        self,
        value,
        mask_trick=False,
        mask_future_timesteps=False,
        incremental_state=None,
    ):
        """Input shape: Time x Batch x Channel
        ` mask_trick` is to use matrix multiplication instead of cumulative sum
         to average the inputs.
         Future timesteps can be masked with the
         `mask_future_timesteps` argument. Padding elements can be excluded from
         the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
         batch x src_len, where padding elements are indicated by 1s.
        """

        assert mask_future_timesteps or incremental_state is None
        if incremental_state is None:
            return self._forward(value, mask_trick, mask_future_timesteps)
        else:
            return self._forward_incremental(
                value, mask_trick, mask_future_timesteps, incremental_state
            )

    def _forward(self, value, mask_trick, mask_future_timesteps):
        length, batch_size = value.size()[:2]
        if not mask_future_timesteps:
            attn = value.mean(dim=0, keepdim=True).repeat(length, 1, 1)
            attn_weights = None
        elif mask_trick:
            v = value.transpose(0, 1)
            attn_weights = torch.arange(1, length + 1, out=v.new(), requires_grad=False)
            attn_weights = (
                attn_weights.reciprocal_().unsqueeze_(1).repeat(1, length).tril(0)
            )
            attn_weights = attn_weights.unsqueeze_(0).repeat(batch_size, 1, 1)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )
            attn = torch.bmm(attn_weights, v)
            attn = attn.transpose(0, 1).contiguous()
        else:
            batch_size = value.size()
            attn_weights = torch.arange(
                1, length + 1, out=value.new(), requires_grad=False
            ).view(length, 1, 1)
            attn = value.cumsum(0) / attn_weights
            attn_weights = None
        return attn, attn_weights

    def _forward_incremental(
        self, value, mask_trick, mask_future_timesteps, incremental_state
    ):
        if mask_trick:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_vec" in saved_state:
                value = torch.cat([saved_state["prev_vec"], value], dim=0)
            saved_state["prev_vec"] = value
            self._set_input_buffer(incremental_state, saved_state)
            attn_weights = None
            attn = value.mean(0, keepdim=True)
        else:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_sum" in saved_state:
                prev_sum = saved_state["prev_sum"]
                if len(prev_sum) == 2:
                    # for tracing, prev_sum does not have sequence axis
                    prev_sum = prev_sum.unsqueeze(0)
                pos = saved_state["prev_pos"] + 1
                curr_sum = prev_sum + value
                attn = curr_sum / pos
            else:
                curr_sum = value
                attn = value
                pos = 1
            saved_state["prev_sum"] = curr_sum
            saved_state["prev_pos"] = pos
            self._set_input_buffer(incremental_state, saved_state)
            attn_weights = None
        return attn, attn_weights

    def extra_repr(self):
        return "embed_dim={}, dropout={}".format(self.embed_dim, self.dropout)


class AverageWindowAttention(AverageAttention):
    """Average attention with window.
    See "Accelerating Neural Transformer via an Average Attention Network"
    for more details.
    """

    def __init__(self, embed_dim, dropout=0.0, bias=True, window_size=0):
        super().__init__(embed_dim, dropout, bias)
        self.window_size = window_size

    def _forward(self, value, mask_trick, mask_future_timesteps):
        if self.window_size == 1:
            return value, None
        length, batch_size = value.size()[:2]
        if not mask_future_timesteps:
            raise NotImplementedError()
        else:
            v = value.transpose(0, 1)
            attn_weights = value.new_ones(length, length, requires_grad=False)
            if self.window_size > 0:
                attn_weights.tril_(0).triu_(1 - self.window_size)
            attn_weights.div_(attn_weights.sum(1, keepdim=True))
            attn_weights = attn_weights.unsqueeze_(0).repeat(batch_size, 1, 1)
            attn = torch.bmm(attn_weights, v)
            attn = attn.transpose(0, 1).contiguous()

        return attn, attn_weights

    def _forward_incremental(
        self, value, mask_trick, mask_future_timesteps, incremental_state
    ):
        if self.window_size == 1:
            return value, None
        length, batch_size = value.size()[:2]
        if mask_trick:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_vec" in saved_state:
                value = torch.cat([saved_state["prev_vec"], value], dim=0)
            saved_state["prev_vec"] = value[-self.window_size :]
            self._set_input_buffer(incremental_state, saved_state)
            attn_weights = None
            attn = value.mean(0, keepdim=True)
        else:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_sum" in saved_state:
                prev_sum = saved_state["prev_sum"]
                values = torch.cat([saved_state["prev_vec"], value], dim=0)
                curr_sum = prev_sum + value
                if values.size(0) > self.window_size:
                    curr_sum -= values[:1]
                avg_size = min(values.size(0), self.window_size)
                attn = curr_sum / avg_size
            else:
                curr_sum = value
                values = value
                attn = value
            saved_state["prev_vec"] = values[-self.window_size :]
            saved_state["prev_sum"] = curr_sum
            self._set_input_buffer(incremental_state, saved_state)
            attn_weights = None
        return attn, attn_weights

    def extra_repr(self):
        return "embed_dim={}, dropout={}, window_size={}".format(
            self.embed_dim, self.dropout, self.window_size
        )
