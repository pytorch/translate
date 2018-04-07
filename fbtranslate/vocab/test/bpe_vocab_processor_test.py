#!/usr/bin/env python3

from libfb.py import testutil
from language_technology.neural_mt.fbtranslate.vocab_processor import (
    VocabProcessor,
)
from language_technology.neural_mt.fbtranslate.vocab.vocab_writer import (
    save_vocab_to_file,
)
import language_technology.neural_mt.fbtranslate.vocab.\
    vocab_constants as constants


class BPEVocabProcessorTest(testutil.BaseFacebookTestCase):

    def generate_tokens_struct(self, tokens):
        return [(token, 0) for token in tokens]

    def test_simple(self):
        vocab_processor = VocabProcessor(save_vocab_to_file(
            vocab_type=constants.BPE_VOCAB_TYPE,
            tokens_and_counts=self.generate_tokens_struct(
                [
                    ',_EOW', '.', 'a', 'as_EOW', 'd',       # 0-4
                    'd_EOW', 'e', 'e_EOW', 'h', 'he_EOW',   # 5-9
                    'l', 'ld_EOW', 'll', 'llo', 'o',        # 10-14
                    'o_EOW', 'or', 'r', 'r_EOW', 's_EOW',   # 15-19
                    't', 'w',                               # 20-21
                ],
            ),
        ))
        self.assertEqual(
            vocab_processor.size(),
            constants.MAX_SPECIAL_TOKENS + 22,
        )
        self.assertEqual(
            vocab_processor.numberize(['the', 'world']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 20,
                    constants.MAX_SPECIAL_TOKENS + 9,
                    constants.MAX_SPECIAL_TOKENS + 21,
                    constants.MAX_SPECIAL_TOKENS + 16,
                    constants.MAX_SPECIAL_TOKENS + 11,
                ],
                [0, 2],
            ),
        )
        self.assertEqual(
            vocab_processor.numberize(['hello', 'world']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 8,
                    constants.MAX_SPECIAL_TOKENS + 6,
                    constants.MAX_SPECIAL_TOKENS + 12,
                    constants.MAX_SPECIAL_TOKENS + 15,
                    constants.MAX_SPECIAL_TOKENS + 21,
                    constants.MAX_SPECIAL_TOKENS + 16,
                    constants.MAX_SPECIAL_TOKENS + 11,
                ],
                [0, 4],
            ),
        )
        # contains full word 'he_EOW'
        self.assertEqual(
            vocab_processor.numberize(['he', 'was', 'told']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 9,
                    constants.MAX_SPECIAL_TOKENS + 21,
                    constants.MAX_SPECIAL_TOKENS + 3,
                    constants.MAX_SPECIAL_TOKENS + 20,
                    constants.MAX_SPECIAL_TOKENS + 14,
                    constants.MAX_SPECIAL_TOKENS + 11,
                ],
                [0, 1, 3],
            ),
        )
        # contains punctuation
        self.assertEqual(
            vocab_processor.numberize(['there', ',', 'we']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 20,
                    constants.MAX_SPECIAL_TOKENS + 8,
                    constants.MAX_SPECIAL_TOKENS + 6,
                    constants.MAX_SPECIAL_TOKENS + 17,
                    constants.MAX_SPECIAL_TOKENS + 7,
                    constants.MAX_SPECIAL_TOKENS + 0,
                    constants.MAX_SPECIAL_TOKENS + 21,
                    constants.MAX_SPECIAL_TOKENS + 7,
                ],
                [0, 5, 6],
            ),
        )
        # Contains UNK '/_EOW'
        self.assertEqual(
            vocab_processor.numberize(['the', 'r/']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 20,
                    constants.MAX_SPECIAL_TOKENS + 9,
                    constants.MAX_SPECIAL_TOKENS + 18,
                ],
                [0, 2],
            ),
        )
        # ensure we choose llo + r_EOW instead of ll + o + r_EOW
        self.assertEqual(
            vocab_processor.numberize(['llor']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 13,
                    constants.MAX_SPECIAL_TOKENS + 18,
                ],
                [0],
            ),
        )
        # make sure the alignment vector has the same number of elements as the
        # input even when a token is deleted. Here, 'z' is an unseen character
        # in the vocab so the token 'z' gets deleted.
        self.assertEqual(
            vocab_processor.numberize(['the', 'z', 'world']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 20,
                    constants.MAX_SPECIAL_TOKENS + 9,
                    constants.MAX_SPECIAL_TOKENS + 21,
                    constants.MAX_SPECIAL_TOKENS + 16,
                    constants.MAX_SPECIAL_TOKENS + 11,
                ],
                [0, 1, 2],
            )
        )
        self.assertEqual(
            vocab_processor.numberize(['the', 'world', 'z']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 20,
                    constants.MAX_SPECIAL_TOKENS + 9,
                    constants.MAX_SPECIAL_TOKENS + 21,
                    constants.MAX_SPECIAL_TOKENS + 16,
                    constants.MAX_SPECIAL_TOKENS + 11,
                ],
                [0, 2, 4],
            )
        )
        self.assertEqual(
            vocab_processor.numberize(['z', 'the', 'world']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 20,
                    constants.MAX_SPECIAL_TOKENS + 9,
                    constants.MAX_SPECIAL_TOKENS + 21,
                    constants.MAX_SPECIAL_TOKENS + 16,
                    constants.MAX_SPECIAL_TOKENS + 11,
                ],
                [0, 0, 2],
            )
        )
        self.assertEqual(
            vocab_processor.denumberize([
                constants.MAX_SPECIAL_TOKENS + 20,
                constants.MAX_SPECIAL_TOKENS + 9,
                constants.MAX_SPECIAL_TOKENS + 21,
                constants.MAX_SPECIAL_TOKENS + 16,
                constants.MAX_SPECIAL_TOKENS + 11,
            ]),
            (['the', 'world'], [0, 2])
        )
        self.assertEqual(
            vocab_processor.denumberize([
                constants.MAX_SPECIAL_TOKENS + 8,
                constants.MAX_SPECIAL_TOKENS + 6,
                constants.MAX_SPECIAL_TOKENS + 12,
                constants.MAX_SPECIAL_TOKENS + 15,
                constants.MAX_SPECIAL_TOKENS + 21,
                constants.MAX_SPECIAL_TOKENS + 16,
                constants.MAX_SPECIAL_TOKENS + 11,
            ]),
            (['hello', 'world'], [0, 4])
        )
        self.assertEqual(
            vocab_processor.get_token(constants.MAX_SPECIAL_TOKENS + 3),
            'as_EOW',
        )
        self.assertEqual(
            vocab_processor.get_token(constants.UNK_ID),
            constants.UNK_TOKEN,
        )
        self.assertEqual(
            vocab_processor.get_token(constants.MAX_SPECIAL_TOKENS + 100),
            None,
        )
