from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from libfb.py import testutil
from language_technology.neural_mt.fbtranslate.vocab_processor import (
    serialize_vocab_thrift_obj_to_file,
    VocabProcessor,
)
from language_technology.neural_mt.vocab import ttypes as ttypes
import language_technology.neural_mt.fbtranslate.vocab.\
    vocab_constants as constants

class MorfessorVocabProcessorTest(testutil.BaseFacebookTestCase):

    def generate_word_struct(self, words):
        return [ttypes.Word(token=word, count=count) for word, count in words]

    def test_simple(self):
        vocab_processor = VocabProcessor(serialize_vocab_thrift_obj_to_file(
            ttypes.BaseVocab(
                morfessor_vocab=ttypes.MorfessorVocab(
                    morfs=self.generate_word_struct(
                        [('their', 10), ('the', 15), ('i', 3), ('r', 2),
                         ('hello', 7), ('hell', 3), ('llo', 2), ('he', 1),
                         ('re', 16), ('there', 1)]
                    ),
                    token_count=60,
                    generate_copy_tokens=False,
                )
            )
        ))
        self.assertEqual(
            vocab_processor.size(),
            constants.MAX_SPECIAL_TOKENS + 10,
        )
        self.assertEqual(
            vocab_processor.numberize(['their', 'the']),
            (
                [
                    constants.START_WORD_ID,
                    constants.MAX_SPECIAL_TOKENS,
                    constants.START_WORD_ID,
                    constants.MAX_SPECIAL_TOKENS + 1,
                ],
                [0, 2],
            ),
        )
        self.assertEqual(
            vocab_processor.numberize(['hello', 'zzz']),
            (
                [
                    constants.START_WORD_ID,
                    constants.MAX_SPECIAL_TOKENS + 4,
                    constants.START_WORD_ID,
                    constants.UNK_ID,
                ],
                [0, 2],
            ),
        )
        self.assertEqual(
            vocab_processor.numberize(['his', 'stir', 'rice']),
            (
                [
                    constants.START_WORD_ID,
                    constants.UNK_ID,
                    constants.MAX_SPECIAL_TOKENS + 2,
                    constants.UNK_ID,
                    constants.START_WORD_ID,
                    constants.UNK_ID,
                    constants.MAX_SPECIAL_TOKENS + 2,
                    constants.MAX_SPECIAL_TOKENS + 3,
                    constants.START_WORD_ID,
                    constants.MAX_SPECIAL_TOKENS + 3,
                    constants.MAX_SPECIAL_TOKENS + 2,
                    constants.UNK_ID,
                ],
                [0, 4, 8],
            ),
        )
        self.assertEqual(
            vocab_processor.numberize(['there']),
            (
                [
                    constants.START_WORD_ID,
                    constants.MAX_SPECIAL_TOKENS + 1,
                    constants.MAX_SPECIAL_TOKENS + 8,
                ],
                [0],
            ),
        )
        self.assertEqual(
            vocab_processor.numberize(['here']),
            (
                [
                    constants.START_WORD_ID,
                    constants.MAX_SPECIAL_TOKENS + 7,
                    constants.MAX_SPECIAL_TOKENS + 8,
                ],
                [0],
            ),
        )
        word_part_ids, word_ids = vocab_processor.numberize(['@DIGITS'])
        digits_tag_id = word_part_ids[1]
        self.assertEqual(
            vocab_processor.denumberize([
                constants.START_WORD_ID,
                constants.MAX_SPECIAL_TOKENS + 7,
                constants.MAX_SPECIAL_TOKENS + 8,
                constants.START_WORD_ID,
                constants.MAX_SPECIAL_TOKENS + 7,
                constants.UNK_ID,
                constants.MAX_SPECIAL_TOKENS + 8,
                constants.START_WORD_ID,
                digits_tag_id,
            ]),
            (['here', constants.UNK_TOKEN, '@DIGITS'], [0, 3, 7])
        )
        # when sequence does not start with START_WORD_TOKEN
        self.assertEqual(
            vocab_processor.denumberize([
                constants.MAX_SPECIAL_TOKENS + 7,
                constants.MAX_SPECIAL_TOKENS + 8,
                constants.START_WORD_ID,
                constants.MAX_SPECIAL_TOKENS + 7,
                constants.UNK_ID,
                constants.MAX_SPECIAL_TOKENS + 8,
                constants.START_WORD_ID,
                digits_tag_id,
            ]),
            (['here', constants.UNK_TOKEN, '@DIGITS'], [0, 2, 6])
        )
        # when sequence does not have START_WORD_TOKEN
        self.assertEqual(
            vocab_processor.denumberize([
                constants.MAX_SPECIAL_TOKENS + 7,
                constants.MAX_SPECIAL_TOKENS + 8,
                constants.MAX_SPECIAL_TOKENS + 7,
                constants.UNK_ID,
                constants.MAX_SPECIAL_TOKENS + 8,
            ]),
            ([constants.UNK_TOKEN], [0])
        )
        self.assertEqual(
            vocab_processor.denumberize([
                constants.START_WORD_ID,
                constants.MAX_SPECIAL_TOKENS + 1,
                constants.START_WORD_ID,
                constants.MAX_SPECIAL_TOKENS + 4,
            ]),
            (['the', 'hello'], [0, 2])
        )
        self.assertEqual(
            vocab_processor.denumberize([
                constants.START_WORD_ID,
                constants.MAX_SPECIAL_TOKENS + 1,
                constants.MAX_SPECIAL_TOKENS + 2,
                constants.MAX_SPECIAL_TOKENS + 3,
                constants.START_WORD_ID,
                constants.MAX_SPECIAL_TOKENS + 7,
                constants.MAX_SPECIAL_TOKENS + 6,
            ]),
            (['their', 'hello'], [0, 4])
        )
        self.assertEqual(
            vocab_processor.get_token(constants.MAX_SPECIAL_TOKENS + 4),
            'hello',
        )
        self.assertEqual(
            vocab_processor.get_token(constants.UNK_ID),
            constants.UNK_TOKEN,
        )
        self.assertEqual(
            vocab_processor.get_token(constants.MAX_SPECIAL_TOKENS + 100),
            None,
        )
