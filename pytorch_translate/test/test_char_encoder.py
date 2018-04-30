#!/usr/bin/env python3

import logging
import numpy as np
import string
import unittest


import torch
from pytorch_translate.char_encoder import CharEmbModel
from pytorch_translate.dictionary import CharDictionary

logger = logging.getLogger(__name__)


class TestCharEncoder(unittest.TestCase):
    def _get_dummy_batch(
        self,
        word_delim_index,
        pad_index,
        bow_index=None,
        eow_index=None,
    ):
        batch_size = 2
        seq_len = [3, 3]
        sent_1 = [[6, 8, 8, 5, 9, 7, 10], [12, 4, 7, 5], [6, 5]]
        sent_2 = [[8, 5, 9], [10, 6, 12, 4], [5, 7, 6, 5]]

        src_tokens = torch.LongTensor([
            sent_1[0]
            + [word_delim_index]
            + sent_1[1]
            + [word_delim_index]
            + sent_1[2],
            [pad_index] * 2
            + sent_2[0]
            + [word_delim_index]
            + sent_2[1]
            + [word_delim_index]
            + sent_2[2],
        ])
        char_batch = np.array([
            [bow_index]
            + sent_1[0]
            + [eow_index],
            [bow_index]
            + sent_1[1]
            + [eow_index]
            + [pad_index] * 3,
            [bow_index]
            + sent_1[2]
            + [eow_index]
            + [pad_index] * 5,
            [bow_index]
            + sent_2[0]
            + [eow_index]
            + [pad_index] * 4,
            [bow_index]
            + sent_2[1]
            + [eow_index]
            + [pad_index] * 3,
            [bow_index]
            + sent_2[2]
            + [eow_index]
            + [pad_index] * 3
        ])

        return src_tokens, batch_size, seq_len, char_batch

    def _dummy_char_vocab(
        self,
        char_token=26,
    ):
        d = CharDictionary()
        for token in list(string.ascii_lowercase):
            d.add_symbol(token)
        d.finalize()
        return d

    def _get_default_model_params(self):
        model_params = {}
        model_params['dictionary'] = self._dummy_char_vocab()
        model_params['char_embed_dim'] = 3
        model_params['word_embed_dim'] = 5
        model_params['convolutions'] = ((7, 2), (8, 3))
        model_params['dropout'] = 0
        return model_params

    def test_conv_step(self):
        model_params = self._get_default_model_params()

        char_encoder = CharEmbModel(**model_params)

        char_input, batch_size, seq_lengths, _ = self._get_dummy_batch(
            char_encoder.dictionary.word_delim_index,
            char_encoder.dictionary.pad_index,
        )
        encoder_inputs = char_encoder(char_input, seq_lengths)
        assert encoder_inputs.size() == torch.Size(
            [batch_size, max(seq_lengths), model_params['word_embed_dim']]
        )

    def test_conv_pool_highway(self):
        model_params = self._get_default_model_params()
        model_params['num_highway_layers'] = 2

        char_encoder = CharEmbModel(**model_params)

        char_input, batch_size, seq_lengths, _ = self._get_dummy_batch(
            char_encoder.dictionary.word_delim_index,
            char_encoder.dictionary.pad_index,
        )
        encoder_inputs = char_encoder(char_input, seq_lengths)
        assert encoder_inputs.size() == torch.Size(
            [batch_size, max(seq_lengths), model_params['word_embed_dim']]
        )

    def test_prepare_char_batch(self):
        model_params = self._get_default_model_params()
        char_encoder = CharEmbModel(**model_params)

        src_tokens, batch_size, seq_lengths, expected_batch = self._get_dummy_batch(
            char_encoder.dictionary.word_delim_index,
            char_encoder.dictionary.pad_index,
            char_encoder.dictionary.bow_index,
            char_encoder.dictionary.eow_index,
        )
        word_batch = char_encoder._prepare_char_batch(
            src_tokens,
            left_padded=True,
        )
        np.testing.assert_array_equal(
            word_batch,
            expected_batch,
        )
