#!/usr/bin/env python3

import unittest

import torch
from pytorch_translate import char_aware_hybrid, transformer
from pytorch_translate.data.dictionary import Dictionary
from pytorch_translate.tasks import pytorch_translate_task as tasks
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
        self.sample = self._dummy_char_data_sample(
            src_dict=self.word_dict,
            dst_dict=self.word_dict,
            src_char_dict=self.char_dict,
            dst_char_dict=self.char_dict,
        )

    def test_forward_training(self):
        """
        We test that if we shuffle the input sample, we will get the same
        forward values, both in training mode (without dropout) and in
        eval mode.
        For the meanwhile, we use an auxiliary hybrid_transformer_rnn
        in order to get the encoder output.
        """
        test_word_decoder_args = test_utils.ModelParamsDict(
            arch="hybrid_transformer_rnn"
        )
        self.task = tasks.DictionaryHolderTask(self.word_dict, self.word_dict)
        word_model = self.task.build_model(test_word_decoder_args)
        word_model.eval()  # Make sure we do not apply dropout.

        test_args = test_utils.ModelParamsDict(arch="char_aware_hybrid")

        decoder_embed_tokens = transformer.build_embedding(
            dictionary=self.word_dict, embed_dim=10
        )
        decoder = char_aware_hybrid.CharAwareHybridRNNDecoder(
            args=test_args,
            src_dict=self.word_dict,
            dst_dict=self.word_dict,
            embed_tokens=decoder_embed_tokens,
            num_chars=len(self.char_dict),
        )

        src_tokens = self.sample["net_input"]["src_tokens"]
        src_lengths = self.sample["net_input"]["src_lengths"]
        prev_output_chars = self.sample["net_input"]["prev_output_chars"][
            :, -1:, :
        ].squeeze(1)
        prev_output_tokens = self.sample["net_input"]["prev_output_tokens"][:, 0:1]

        encoder_out = word_model.encoder(src_tokens, src_lengths)

        embed_output = decoder._embed_prev_outputs(
            prev_output_tokens=prev_output_tokens, prev_output_chars=prev_output_chars
        )[0]
        forward_output = decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            prev_output_chars=prev_output_chars,
        )
        output_logits = forward_output[0]

        prev_output_tokens_shuffled = torch.cat(
            [prev_output_tokens[1:], prev_output_tokens[0].unsqueeze(0)], dim=0
        )
        prev_output_chars_shuffled = torch.cat(
            [prev_output_chars[1:], prev_output_chars[0].unsqueeze(0)], dim=0
        )
        src_tokens_shuffled = torch.cat(
            [src_tokens[1:], src_tokens[0].unsqueeze(0)], dim=0
        )

        # Making sure shuffling is done correctly.
        assert torch.equal(src_tokens[0], src_tokens_shuffled[2])
        assert torch.equal(src_tokens[1], src_tokens_shuffled[0])
        assert torch.equal(src_tokens[2], src_tokens_shuffled[1])
        assert torch.equal(prev_output_chars[0], prev_output_chars_shuffled[2])
        assert torch.equal(prev_output_chars[1], prev_output_chars_shuffled[0])
        assert torch.equal(prev_output_chars[2], prev_output_chars_shuffled[1])
        assert torch.equal(prev_output_tokens[0], prev_output_tokens_shuffled[2])
        assert torch.equal(prev_output_tokens[1], prev_output_tokens_shuffled[0])
        assert torch.equal(prev_output_tokens[2], prev_output_tokens_shuffled[1])

        # Making sure that we embed the inputs correctly.
        encoder_out_shuffled = word_model.encoder(src_tokens_shuffled, src_lengths)
        embed_output_shuffled = decoder._embed_prev_outputs(
            prev_output_tokens=prev_output_tokens_shuffled,
            prev_output_chars=prev_output_chars_shuffled,
        )[0]
        assert embed_output[0, 0].equal(embed_output_shuffled[0, 2])
        assert embed_output[0, 1].equal(embed_output_shuffled[0, 0])
        assert embed_output[0, 2].equal(embed_output_shuffled[0, 1])

        # Making sure the output of the forward function is correct.
        forward_output_shuffled = decoder(
            prev_output_tokens=prev_output_tokens_shuffled,
            encoder_out=encoder_out_shuffled,
            prev_output_chars=prev_output_chars_shuffled,
        )
        output_logits_shuffled = forward_output_shuffled[0]

        assert encoder_out[0][:, 0, :].equal(encoder_out_shuffled[0][:, 2, :])
        assert encoder_out[0][:, 1, :].equal(encoder_out_shuffled[0][:, 0, :])
        assert encoder_out[0][:, 2, :].equal(encoder_out_shuffled[0][:, 1, :])

        assert output_logits[0].equal(output_logits_shuffled[2])
        assert output_logits[1].equal(output_logits_shuffled[0])
        assert output_logits[2].equal(output_logits_shuffled[1])

        """
        Now trying in the eval mode.
        """
        decoder.eval()
        forward_output = decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            prev_output_chars=prev_output_chars,
        )
        output_logits = forward_output[0]
        forward_output_shuffled = decoder(
            prev_output_tokens=prev_output_tokens_shuffled,
            encoder_out=encoder_out_shuffled,
            prev_output_chars=prev_output_chars_shuffled,
        )
        output_logits_shuffled = forward_output_shuffled[0]
        assert output_logits[0].equal(output_logits_shuffled[2])
        assert output_logits[1].equal(output_logits_shuffled[0])
        assert output_logits[2].equal(output_logits_shuffled[1])

    def _dummy_char_data_sample(
        self,
        src_dict: Dictionary,
        dst_dict: Dictionary,
        src_char_dict: Dictionary,
        dst_char_dict: Dictionary,
        batch_size: int = 3,
        input_seq_length: int = 5,
        output_seq_length: int = 4,
        max_word_length: int = 7,
    ):
        src_tokens = torch.randint(
            low=src_dict.nspecial,
            high=len(src_dict),
            size=(batch_size, input_seq_length),
        ).long()

        output_sequence = torch.randint(
            low=dst_dict.nspecial,
            high=len(dst_dict),
            size=(batch_size, output_seq_length),
        ).long()
        eos_column = torch.LongTensor([dst_dict.eos_index] * batch_size).unsqueeze(1)
        prev_output_tokens = torch.cat([eos_column, output_sequence], dim=1)
        target_tokens = torch.cat([output_sequence, eos_column], dim=1)

        src_char_inds = torch.randint(
            low=src_char_dict.nspecial,
            high=len(src_char_dict),
            size=(batch_size, output_seq_length, max_word_length),
        ).long()

        eos_char_column = (
            torch.LongTensor(batch_size, output_seq_length)
            .fill_(dst_char_dict.eos_index)
            .unsqueeze(2)
        )

        tgt_char_inds = torch.randint(
            low=dst_char_dict.nspecial,
            high=len(dst_char_dict),
            size=(batch_size, output_seq_length, max_word_length),
        ).long()
        prev_output_chars = torch.cat([eos_char_column, tgt_char_inds], dim=2)
        word_lengths = torch.LongTensor(batch_size, input_seq_length).fill_(
            max_word_length
        )
        prev_tgt_word_lengths = torch.cat(
            (torch.ones((batch_size, 1), dtype=torch.long), word_lengths), dim=1
        )

        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "prev_output_tokens": prev_output_tokens,
                "src_lengths": torch.LongTensor([input_seq_length] * batch_size),
                "char_inds": src_char_inds,
                "word_lengths": word_lengths,
                "prev_output_chars": prev_output_chars,
                "prev_output_word_lengths": prev_tgt_word_lengths,
            },
            "target": target_tokens,
            "ntokens": target_tokens.numel(),
            "target_char_inds": tgt_char_inds,
            "tgt_word_lengths": word_lengths,
            "weights": None,
        }
        return sample
