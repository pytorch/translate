#!/usr/bin/env python3

import logging
import numpy as np
import unittest

from caffe2.python import attention as caffe2attention
from caffe2.python.core import workspace
from caffe2.python.models.seq2seq.seq2seq_model_helper import Seq2SeqModelHelper

import torch
from torch import autograd
from language_technology.neural_mt.fbtranslate.rnn import AttentionLayer

logger = logging.getLogger(__name__)


class TestAttentionLayer(unittest.TestCase):
    def _test_attention_base(
        self,
        attention_type,
        decoder_hidden_state_dim,
        encoder_output_dim,
        batch_size,
        src_sentence_length,
        numpy_reference_fn,
        caffe2_reference_fn,
    ):
        d_source_hids = [src_sentence_length, batch_size, encoder_output_dim]
        d_input = [batch_size, decoder_hidden_state_dim]
        input = np.random.rand(*d_input).astype(np.float32)
        input_var = autograd.Variable(torch.FloatTensor(input))

        source_hids = np.random.rand(*d_source_hids).astype(np.float32)
        source_hids_var = autograd.Variable(torch.FloatTensor(source_hids))

        attention = AttentionLayer(
            decoder_hidden_state_dim,
            encoder_output_dim,
            attention_type
        )
        out, attn_scores = attention(
            input_var,
            source_hids_var,
            autograd.Variable(
                torch.LongTensor([src_sentence_length] * batch_size),
            ),
        )
        out_ref, attn_scores_ref = numpy_reference_fn(
            input,
            source_hids,
            decoder_hidden_state_dim,
            encoder_output_dim,
            attention,
        )
        np.testing.assert_allclose(out_ref, out.data.numpy(), atol=1e-5)
        np.testing.assert_allclose(
            attn_scores_ref,
            attn_scores.data.numpy(),
            atol=1e-5,
        )

        out_ref, attn_scores_ref, _ = caffe2_reference_fn(
            input,
            source_hids,
            decoder_hidden_state_dim,
            encoder_output_dim,
            attention,
        )
        np.testing.assert_allclose(out_ref, out.data.numpy(), atol=1e-5)
        np.testing.assert_allclose(
            attn_scores_ref,
            attn_scores.data.numpy(),
            atol=1e-5,
        )

    def _softmax_ref(self, w):
        maxes = np.amax(w, axis=-1, keepdims=True)
        e = np.exp(w - maxes)
        dist = e / np.sum(e, axis=-1, keepdims=True)
        return dist

    def _dot_attention_numpy_ref(
        self,
        input,
        source_hids,
        decoder_hidden_state_dim,
        encoder_output_dim,
        attention,
    ):
        if decoder_hidden_state_dim != encoder_output_dim:
            # (encoder_output_dim, decoder_hidden_state_dim)
            input_proj_w = attention.input_proj.weight.data.numpy()

            # (1, encoder_output_dim)
            input_proj_b = attention.input_proj.bias.data.numpy()
            input_proj_b = input_proj_b[np.newaxis, :]

            # (batch_size, encoder_output_dim)
            input = np.matmul(input, np.transpose(input_proj_w)) + input_proj_b

        # (srclen, batch_size, encoder_output_dim)
        attn_scores = source_hids * input[np.newaxis, :, :]

        # (srclen, batch_size)
        attn_scores = np.sum(attn_scores, axis=2)
        attn_scores = np.transpose(self._softmax_ref(np.transpose(attn_scores)))

        # (srclen, batch_size, encoder_output_dim)
        attn_weighted_context = attn_scores[:, :, np.newaxis] * source_hids
        # (batch_size, encoder_output_dim)
        attn_weighted_context = np.sum(attn_weighted_context, axis=0)

        return attn_weighted_context, attn_scores

    def _dot_attention_caffe2_ref(
        self,
        input,
        source_hids,
        decoder_hidden_state_dim,
        encoder_output_dim,
        attention,
    ):
        model = Seq2SeqModelHelper()
        encoder_outputs_transposed = np.transpose(source_hids, axes=[1, 2, 0])
        decoder_hidden_state_t = input[np.newaxis, :, :]

        workspace.FeedBlob(
            'encoder_outputs_transposed',
            encoder_outputs_transposed,
        )
        workspace.FeedBlob('decoder_hidden_state_t', decoder_hidden_state_t)

        test_scope = 'pytorch_attention_test'

        if decoder_hidden_state_dim != encoder_output_dim:
            # (encoder_output_dim, decoder_hidden_state_dim)
            input_proj_w = attention.input_proj.weight.data.numpy()
            # (1, encoder_output_dim)
            input_proj_b = attention.input_proj.bias.data.numpy()
            workspace.FeedBlob(
                '{}/weighted_decoder_hidden_state_w'.format(test_scope),
                input_proj_w,
            )
            workspace.FeedBlob(
                '{}/weighted_decoder_hidden_state_b'.format(test_scope),
                input_proj_b,
            )

        (
            attention_weighted_encoder_context,
            attention_weights_3d,
            _,
        ) = caffe2attention.apply_dot_attention(
            model,
            encoder_output_dim,
            'encoder_outputs_transposed',
            'decoder_hidden_state_t',
            decoder_hidden_state_dim,
            scope=test_scope,
        )

        workspace.RunNetOnce(model.net)

        # (1, batch_size, encoder_output_dim)
        attn_weighted_context = workspace.FetchBlob(
            attention_weighted_encoder_context,
        )
        attn_weighted_context = np.squeeze(attn_weighted_context)

        # (srclen, batch_size)
        attn_weights = workspace.FetchBlob(attention_weights_3d)
        attn_weights = np.transpose(np.squeeze(attn_weights))
        return attn_weighted_context, attn_weights, _

    def test_dot_attention_different_dims(self):
        self._test_attention_base(
            attention_type='dot',
            decoder_hidden_state_dim=10,
            encoder_output_dim=12,
            batch_size=4,
            src_sentence_length=11,
            numpy_reference_fn=self._dot_attention_numpy_ref,
            caffe2_reference_fn=self._dot_attention_caffe2_ref,
        )

    def test_dot_attention_same_dims(self):
        self._test_attention_base(
            attention_type='dot',
            decoder_hidden_state_dim=10,
            encoder_output_dim=10,
            batch_size=4,
            src_sentence_length=11,
            numpy_reference_fn=self._dot_attention_numpy_ref,
            caffe2_reference_fn=self._dot_attention_caffe2_ref,
        )
