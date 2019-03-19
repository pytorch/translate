#!/usr/bin/env python3

import random
import unittest

import numpy as np
import pytorch_translate.attention.multihead_attention as multihead_attention
import torch
from pytorch_translate.attention import attention_utils, dot_attention, mlp_attention


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


def _softmax(x):  # softmax over 4 dim matrix
    """ Numpy-based reference softmax over 4 dim matrix"""
    output = np.zeros(x.shape, dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x_curr = x[i, j, k, :]
                e_x = np.exp(x_curr - np.amax(x_curr))
                output[i, j, k, :] = e_x / np.sum(e_x)
    return output


def _batchmatmul(a, b):  # batchmatmul over 4 dim matrix
    """ Numpy-based batch matrix multiply over 4 dim matrix"""
    assert a.shape[0] == b.shape[0]
    assert a.shape[1] == b.shape[1]
    retval = np.zeros(
        (a.shape[0], a.shape[1], a.shape[2], b.shape[3]), dtype=np.float32
    )
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            retval[i, j, :, :] = np.matmul(a[i, j, :, :], b[i, j, :, :])
    return retval


class MultiheadAttentionTest(unittest.TestCase):
    def _scaled_dot_attn_ref(self, Q, K, V, dims, unseen_mask=False, src_lengths=None):
        """ Numpy-based reference implementation of scaled dot attention
        for testing"""
        QKT = _batchmatmul(
            Q,
            np.transpose(K, axes=[0, 1, 3, 2])
            / np.sqrt(dims[3], dtype=np.float32),  # divide by sqrt(d_head)
        )
        if unseen_mask or src_lengths is not None:
            b1, b2, s1, s2 = QKT.shape
            # assert s1 == s2
            for i in range(b1):
                for j in range(b2):
                    for m in range(s1):
                        for n in range(s2):
                            if unseen_mask and n > m:
                                QKT[i, j, m, n] = -np.inf
                            if src_lengths is not None and n >= src_lengths[i]:
                                QKT[i, j, m, n] = -np.inf
        reference = _softmax(QKT)
        reference = _batchmatmul(reference, V)
        return reference

    def _generate_src_lengths(self, batch_size, seq_len):
        src_lengths = np.array([random.randint(1, seq_len) for i in range(batch_size)])

        # max source length has to equal seq_len, so randomly choose
        # one example to have source length = seq_len
        max_len_example_i = random.randint(0, batch_size - 1)
        src_lengths[max_len_example_i] = seq_len

        src_lengths_tensor = torch.from_numpy(src_lengths).int()
        return src_lengths, src_lengths_tensor

    def _split_heads_ref(self, X, dims, nheads, d_head):
        X_split = np.reshape(X, dims[:2] + [nheads, d_head])
        X_split_transposed = np.transpose(X_split, [0, 2, 1, 3])
        reference = np.reshape(X_split_transposed, [dims[0], nheads, dims[1], d_head])
        return reference

    def _combine_heads_ref(self, X, dims, nheads, d_head):
        X_transposed = np.transpose(X, [0, 2, 1, 3])
        reference = np.reshape(X_transposed, dims[:2] + [nheads * d_head])
        return reference

    def _fc(self, X, X_name, module, start=None, end=None):
        X_fc_b = None
        X_fc_w = None
        for name, param in module.named_parameters():
            if X_name + "weight" in name:
                if X_fc_w is not None:
                    raise Exception(f"Duplicate FC name {name} found")
                X_fc_w = param[start:end, :].detach().numpy()
            elif X_name + "bias" in name:
                if X_fc_b is not None:
                    raise Exception(f"Duplicate FC name {name} found")
                X_fc_b = param[start:end].detach().numpy()
        return np.matmul(X, np.transpose(X_fc_w)) + X_fc_b

    def _multihead_attn_test_helper(self, use_src_lengths):
        for _ in range(100):
            batch_sz, seq_len = [random.randint(2, 10) for r in range(2)]
            d_head = random.randint(3, 10)
            nheads = random.randint(3, 10)
            d_model = d_head * nheads
            dims = [batch_sz, seq_len, d_model]

            src_lengths = None
            src_lengths_tensor = None
            if use_src_lengths:
                src_lengths, src_lengths_tensor = self._generate_src_lengths(
                    batch_size=batch_sz, seq_len=seq_len
                )

            decoder_state = np.random.rand(batch_sz, d_model).astype(np.float32)
            K = np.random.rand(*dims).astype(np.float32)
            V = K
            Q = np.expand_dims(decoder_state, 1)

            decoder_state_tensor = torch.from_numpy(decoder_state).float()
            source_hid_tensor = torch.from_numpy(K).float().transpose(0, 1)

            multihead_attn_module = multihead_attention.MultiheadAttention(
                context_dim=d_model, decoder_hidden_state_dim=d_model, nheads=nheads
            )

            result = (
                multihead_attn_module(
                    decoder_state=decoder_state_tensor,
                    source_hids=source_hid_tensor,
                    src_lengths=src_lengths_tensor,
                )[0]
                .detach()
                .numpy()
            )

            Q_fc = self._fc(Q, "in_proj_", multihead_attn_module, end=d_model)
            K_fc = self._fc(
                K, "in_proj_", multihead_attn_module, start=d_model, end=2 * d_model
            )
            V_fc = self._fc(V, "in_proj_", multihead_attn_module, start=2 * d_model)

            Q_split = self._split_heads_ref(
                Q_fc, [batch_sz, 1, d_model], nheads, d_head
            )
            K_split = self._split_heads_ref(K_fc, dims, nheads, d_head)
            V_split = self._split_heads_ref(V_fc, dims, nheads, d_head)

            attn_heads = self._scaled_dot_attn_ref(
                Q=Q_split,
                K=K_split,
                V=V_split,
                dims=Q_split.shape,
                src_lengths=src_lengths,
            )

            combined_attn_heads = self._combine_heads_ref(
                X=attn_heads, dims=[batch_sz, 1], nheads=nheads, d_head=d_head
            )

            reference = self._fc(
                combined_attn_heads, "out_proj.", multihead_attn_module
            )
            reference = np.squeeze(reference, axis=1)
            self.assertEqual(tuple(result.shape), (batch_sz, d_model))
            np.testing.assert_allclose(result, reference, atol=1e-5)

    def test_multihead_attn_no_masking(self):
        self._multihead_attn_test_helper(use_src_lengths=None)

    def test_multihead_attn_with_src_lengths(self):
        self._multihead_attn_test_helper(use_src_lengths=True)
