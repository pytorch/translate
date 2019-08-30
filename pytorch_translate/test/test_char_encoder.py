#!/usr/bin/env python3

import unittest

import torch
from pytorch_translate import char_encoder
from pytorch_translate.data.dictionary import Dictionary
from pytorch_translate.test import utils as test_utils


class TestCharAwareHybrid(unittest.TestCase):
    def setUp(self):
        self.src_txt, self.trg_txt = test_utils.create_test_text_files()
        self.vocab_file_path = test_utils.make_temp_file()
        self.word_dict = Dictionary.build_vocab_file(
            corpus_files=[self.src_txt, self.trg_txt],
            vocab_file=self.vocab_file_path,
            max_vocab_size=0,
            padding_factor=1,  # don't add extra padding symbols
        )
        self.char_dict = Dictionary.build_vocab_file(
            corpus_files=[self.src_txt, self.trg_txt],
            vocab_file=self.vocab_file_path,
            max_vocab_size=0,
            is_char_vocab=True,
            padding_factor=1,  # don't add extra padding symbols
        )

    def test_padding_sanity(self):
        """
        We test that if we pad the same char_sequence differently we get
        the same result.
        """
        bsz, max_chars = 3, 8
        encoder = char_encoder.CharCNNModel(
            dictionary=self.word_dict, num_chars=110, char_embed_dim=32
        )
        encoder.eval()

        # First getting values without standard padding.
        src_char_inds = torch.randint(
            low=100, high=110, size=(bsz, 1, max_chars)
        ).long()

        # Put deterministic pads
        src_char_inds[1, 0, max_chars - 1] = 0
        src_char_inds[1, 0, max_chars - 2] = 0
        src_char_inds[2, 0, max_chars - 1] = 0

        char_inds_flat = src_char_inds.view(-1, max_chars).t()
        char_cnn_output = encoder(char_inds_flat)

        # First testing if the same data that is chunked to smaller subparts
        # gets the same output.
        src_char_inds_parts = [src_char_inds[i] for i in range(bsz)]
        char_cnn_output_parts = []
        for i in range(bsz):
            char_inds_flat = src_char_inds_parts[i].view(-1, max_chars).t()
            char_cnn_output_part = encoder(char_inds_flat)
            char_cnn_output_parts.append(char_cnn_output_part)
        char_cnn_output_parts = torch.cat(char_cnn_output_parts)
        assert torch.equal(char_cnn_output, char_cnn_output_parts)

        # Asserting if we get the same result with one level of padding.
        one_pad_inds = torch.Tensor(bsz, 1, 1).long().fill_(self.word_dict.pad_index)
        src_char_inds_one_pad = torch.cat([src_char_inds, one_pad_inds], dim=2)
        char_inds_one_pad_flat = src_char_inds_one_pad.view(-1, max_chars + 1).t()
        char_cnn_one_pad_output = encoder(char_inds_one_pad_flat)
        assert torch.allclose(char_cnn_output, char_cnn_one_pad_output)
        assert torch.equal(char_cnn_output, char_cnn_one_pad_output)
