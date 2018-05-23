#!/usr/bin/env python3

import numpy as np
import random
import unittest

import torch

import language_technology.neural_mt.os.pytorch_translate.research.\
    multihead_attention as multihead_attention


def _softmax(x):  # softmax over 4 dim matrix
    output = np.zeros(x.shape, dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x_curr = x[i, j, k, :]
                print(np.amax(x_curr))
                e_x = np.exp(x_curr - np.amax(x_curr))
                output[i, j, k, :] = e_x / np.sum(e_x)
    return output


def _batchmatmul(a, b):  # batchmatmul over 4 dim matrix
    assert a.shape[0] == b.shape[0]
    assert a.shape[1] == b.shape[1]
    retval = np.zeros(
        (a.shape[0], a.shape[1], a.shape[2], b.shape[3]),
        dtype=np.float32,
    )
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            retval[i, j, :, :] = np.matmul(a[i, j, :, :], b[i, j, :, :])
    return retval


class MultiheadAttentionTest(unittest.TestCase):
    def _scaled_dot_attn_ref(
        self,
        Q,
        K,
        V,
        dims,
        unseen_mask=False,
        src_lengths=None,
    ):
        QKT = _batchmatmul(
            Q,
            np.transpose(K, axes=[0, 1, 3, 2]) /
            np.sqrt(dims[3], dtype=np.float32)  # divide by sqrt(d_head)
        )
        if unseen_mask or src_lengths is not None:
            b1, b2, s1, s2 = QKT.shape
            assert s1 == s2
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
        src_lengths = np.array([
            random.randint(1, seq_len) for i in range(batch_size)
        ])

        # max source length has to equal seq_len, so randomly choose
        # one example to have source length = seq_len
        max_len_example_i = random.randint(0, batch_size - 1)
        src_lengths[max_len_example_i] = seq_len

        src_lengths_tensor = torch.from_numpy(src_lengths).int()
        return src_lengths, src_lengths_tensor

    def _scaled_dot_attn_test_helper(self, unseen_mask, use_src_lengths):
        for _ in range(100):
            # (batch size, nheads, sequence length, d_head)
            dims = [random.randint(2, 10) for r in range(4)]
            Q = np.random.rand(*dims).astype(np.float32)
            K = np.random.rand(*dims).astype(np.float32)
            V = np.random.rand(*dims).astype(np.float32)

            src_lengths = None
            src_lengths_tensor = None
            if use_src_lengths:
                src_lengths, src_lengths_tensor = self._generate_src_lengths(
                    batch_size=dims[0],
                    seq_len=dims[2],
                )

            Q_tensor = torch.from_numpy(Q).float()
            K_tensor = torch.from_numpy(K).float()
            V_tensor = torch.from_numpy(V).float()

            result = multihead_attention.scaled_dot_prod_attn(
                query=Q_tensor,
                key=K_tensor,
                value=V_tensor,
                unseen_mask=unseen_mask,
                src_lengths=src_lengths_tensor,
            )[0].data.numpy()

            reference = self._scaled_dot_attn_ref(
                Q=Q,
                K=K,
                V=V,
                dims=dims,
                unseen_mask=unseen_mask,
                src_lengths=src_lengths,
            )

            np.testing.assert_allclose(result, reference, atol=1e-5)

    def test_scaled_dot_attn_no_masking(self):
        self._scaled_dot_attn_test_helper(
            unseen_mask=False,
            use_src_lengths=None,
        )

    def test_scaled_dot_attn_with_unseen_mask(self):
        self._scaled_dot_attn_test_helper(
            unseen_mask=True,
            use_src_lengths=None,
        )

    def test_scaled_dot_attn_with_src_lengths(self):
        self._scaled_dot_attn_test_helper(
            unseen_mask=True,
            use_src_lengths=True,
        )

    def test_scaled_dot_attn_both_masks(self):
        self._scaled_dot_attn_test_helper(
            unseen_mask=True,
            use_src_lengths=True,
        )

    def _split_heads_ref(self, X, dims, nheads, d_head):
        X_split = np.reshape(X, dims[:2] + [nheads, d_head])
        X_split_transposed = np.transpose(X_split, [0, 2, 1, 3])
        reference = np.reshape(
            X_split_transposed, [dims[0], nheads, dims[1], d_head]
        )
        return reference

    def test_split_heads(self):
        for _ in range(100):

            batch_and_seq_len = [random.randint(2, 10) for r in range(2)]
            d_head = random.randint(3, 10)
            nheads = random.randint(3, 10)
            d_model = d_head * nheads
            dims = batch_and_seq_len + [d_model]
            X = np.random.rand(*dims).astype(np.float32)

            X_tensor = torch.from_numpy(X).float()

            result = multihead_attention.split_heads(X_tensor, nheads).numpy()

            reference = self._split_heads_ref(X, dims, nheads, d_head)
            np.testing.assert_allclose(result, reference, atol=1e-5)

    def _combine_heads_ref(self, X, dims, nheads, d_head):
        X_transposed = np.transpose(X, [0, 2, 1, 3])
        reference = np.reshape(X_transposed, dims[:2] + [nheads * d_head])
        return reference

    def test_combine_heads(self):
        for _ in range(100):
            batch_and_seq_len = [random.randint(2, 10) for r in range(2)]
            d_head = random.randint(3, 10)
            nheads = random.randint(3, 10)
            d_model = d_head * nheads
            dims = batch_and_seq_len + [d_model]
            X = np.random.rand(*dims).astype(np.float32)

            X_split_heads = self._split_heads_ref(X, dims, nheads, d_head)
            X_split_heads_tensor = torch.from_numpy(X_split_heads).float()
            result = multihead_attention.combine_heads(X_split_heads_tensor)

            reference = self._combine_heads_ref(X_split_heads, dims, nheads, d_head)

            # Makes sure combine_heads is an exact inverse of split_heads_output
            np.testing.assert_allclose(result, X, atol=1e-5)
            np.testing.assert_allclose(result, reference, atol=1e-5)

    def _fc(self, X, X_name, module):
        X_fc_b = 0
        for name, param in module.named_parameters():
            if X_name + '_fc.weight' in name:
                X_fc_w = param.data.numpy()
            elif X_name + '_fc.bias' in name:
                X_fc_b = param.data.numpy()
        return np.matmul(X, np.transpose(X_fc_w)) + X_fc_b

    def _multihead_attn_test_helper(self, unseen_mask, use_src_lengths):
        for _ in range(100):
            batch_and_seq_len = [random.randint(2, 10) for r in range(2)]
            d_head = random.randint(3, 10)
            nheads = random.randint(3, 10)
            d_model = d_head * nheads
            dims = batch_and_seq_len + [d_model]

            src_lengths = None
            src_lengths_tensor = None
            if use_src_lengths:
                src_lengths, src_lengths_tensor = self._generate_src_lengths(
                    batch_size=batch_and_seq_len[0],
                    seq_len=batch_and_seq_len[1],
                )

            Q = np.random.rand(*dims).astype(np.float32)
            K = np.random.rand(*dims).astype(np.float32)
            V = np.random.rand(*dims).astype(np.float32)

            Q_tensor = torch.from_numpy(Q).float()
            K_tensor = torch.from_numpy(K).float()
            V_tensor = torch.from_numpy(V).float()

            multihead_attn_module = multihead_attention.MultiheadAttention(
                nheads=nheads,
                d_model=d_model,
            )

            result = multihead_attn_module(
                query=Q_tensor,
                key=K_tensor,
                value=V_tensor,
                unseen_mask=unseen_mask,
                src_lengths=src_lengths_tensor,
            ).detach().numpy()

            Q_fc = self._fc(Q, 'Q', multihead_attn_module)
            K_fc = self._fc(K, 'K', multihead_attn_module)
            V_fc = self._fc(V, 'V', multihead_attn_module)

            Q_split = self._split_heads_ref(Q_fc, dims, nheads, d_head)
            K_split = self._split_heads_ref(K_fc, dims, nheads, d_head)
            V_split = self._split_heads_ref(V_fc, dims, nheads, d_head)

            attn_heads = self._scaled_dot_attn_ref(
                Q=Q_split,
                K=K_split,
                V=V_split,
                dims=Q_split.shape,
                unseen_mask=unseen_mask,
                src_lengths=src_lengths,
            )

            combined_attn_heads = self._combine_heads_ref(
                X=attn_heads,
                dims=dims,
                nheads=nheads,
                d_head=d_head,
            )

            reference = self._fc(
                combined_attn_heads,
                'output',
                multihead_attn_module,
            )

            np.testing.assert_allclose(result, reference, atol=1e-5)

    def test_multihead_attn_no_masking(self):
        self._multihead_attn_test_helper(
            unseen_mask=False,
            use_src_lengths=None,
        )

    def test_multihead_attn_with_unseen_mask(self):
        self._multihead_attn_test_helper(
            unseen_mask=True,
            use_src_lengths=None,
        )

    def test_multihead_attn_with_src_lengths(self):
        self._multihead_attn_test_helper(
            unseen_mask=True,
            use_src_lengths=True,
        )

    def test_multihead_attn_both_masks(self):
        self._multihead_attn_test_helper(
            unseen_mask=True,
            use_src_lengths=True,
        )
