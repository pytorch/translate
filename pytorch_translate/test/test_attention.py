#!/usr/bin/env python3

import unittest
import torch

from pytorch_translate.attention import attention_utils

from pytorch_translate.attention import dot_attention
from pytorch_translate.attention import mlp_attention


class TestAttention(unittest.TestCase):

    def setUp(self):
        self.bsz = 10
        self.src_len = 5
        self.ctx_dim = 3
        self.dec_dim = 4
        self.att_dim = 2

    def test_masked_softmax(self):
        scores = torch.rand(20, 20)
        lengths = torch.arange(start=1, end=21)

        masked_normalized_scores = attention_utils.masked_softmax(
            scores, lengths, src_length_masking=True
        )

        for i in range(20):
            scores_sum = masked_normalized_scores[i].numpy().sum()
            self.assertAlmostEqual(scores_sum, 1, places=6)

    def _test_attention(self, attention):
        dummy_source_hids = torch.rand(self.src_len, self.bsz, self.ctx_dim)
        dummy_decoder_state = torch.rand(self.bsz, self.dec_dim)
        dummy_src_lengths = torch.fmod(torch.arange(self.bsz), self.src_len) + 1
        attention(dummy_decoder_state, dummy_source_hids, dummy_src_lengths)

    def test_dot_attention(self):
        self._test_attention(
            dot_attention.DotAttention(
                self.dec_dim,
                self.ctx_dim,
                src_length_masking=True,
                force_projection=True,
            )
        )

    def test_mlp_attention(self):
        self._test_attention(
            mlp_attention.MLPAttention(
                self.dec_dim,
                self.ctx_dim,
                src_length_masking=True,
                attention_dim=self.att_dim,
            )
        )
