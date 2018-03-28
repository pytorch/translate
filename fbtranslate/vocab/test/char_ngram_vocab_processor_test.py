from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile

from libfb.py import testutil
from language_technology.neural_mt.fbtranslate.vocab_processor import (
    serialize_vocab_thrift_obj_to_file,
    VocabProcessor,
)
from language_technology.neural_mt.vocab import ttypes as ttypes
import language_technology.neural_mt.fbtranslate.vocab.\
    vocab_constants as constants

class CharNgramVocabProcessorTest(testutil.BaseFacebookTestCase):

    def generate_word_struct(self, words):
        return [ttypes.Word(token=word, count=0) for word in words]

    def test_simple(self):
        vocab_processor = VocabProcessor(serialize_vocab_thrift_obj_to_file(
            ttypes.BaseVocab(
                char_ngram_vocab=ttypes.CharNgramVocab(
                    words=self.generate_word_struct(['cat', 'c', 'a', 'r']),
                    ngrams=self.generate_word_struct(
                        ['c', 'a', 'r', 't'],
                    ),
                    ngram_size=1,
                    generate_copy_tokens=False,
                )
            )
        ))
        self.assertEqual(
            vocab_processor.numberize(['cat', 'car']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS,
                    constants.START_WORD_ID,
                    constants.MAX_SPECIAL_TOKENS + 4,
                    constants.MAX_SPECIAL_TOKENS + 5,
                    constants.MAX_SPECIAL_TOKENS + 6,
                ],
                [0, 1],
            ),
        )
        self.assertEqual(
            vocab_processor.denumberize(
                vocab_processor.numberize(['cat', 'car'])[0],
            ),
            (['cat', 'car'], [0, 1])
        )
        self.assertEqual(
            [vocab_processor.get_id(token.decode("utf-8")) for token
                in vocab_processor.tokenize(["cart"])[0]],
            vocab_processor.numberize(["cart"])[0]
        )

        vocab_processor = VocabProcessor(serialize_vocab_thrift_obj_to_file(
            ttypes.BaseVocab(
                char_ngram_vocab=ttypes.CharNgramVocab(
                    words=self.generate_word_struct(['.', 'the', ',', 'of']),
                    ngrams=self.generate_word_struct(
                        ['e', 'th', '.', 'in', '\uffe4\u5c8d', '\uffe4\u9593'],
                    ),
                    ngram_size=2,
                    generate_copy_tokens=False,
                )
            )
        ))
        self.assertEqual(
            vocab_processor.size(),
            constants.MAX_SPECIAL_TOKENS + 10,
        )
        self.assertEqual(
            vocab_processor.numberize(['the', 'thin', 'the']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 1,
                    constants.START_WORD_ID,
                    constants.MAX_SPECIAL_TOKENS + 5,
                    constants.MAX_SPECIAL_TOKENS + 7,
                    constants.MAX_SPECIAL_TOKENS + 1,
                ],
                [0, 1, 4],
            ),
        )
        self.assertEqual(
            vocab_processor.numberize(['the', 'of']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 1,
                    constants.MAX_SPECIAL_TOKENS + 3,
                ],
                [0, 1],
            ),
        )
        self.assertEqual(
            vocab_processor.numberize(['the', 'inine']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 1,
                    constants.START_WORD_ID,
                    constants.MAX_SPECIAL_TOKENS + 7,
                    constants.MAX_SPECIAL_TOKENS + 7,
                    constants.MAX_SPECIAL_TOKENS + 4,
                ],
                [0, 1],
            ),
        )
        self.assertEqual(
            vocab_processor.numberize(['the', 'th\uffe4\u5c8dine']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 1,
                    constants.START_WORD_ID,
                    constants.MAX_SPECIAL_TOKENS + 5,
                    constants.MAX_SPECIAL_TOKENS + 8,
                    constants.MAX_SPECIAL_TOKENS + 7,
                    constants.MAX_SPECIAL_TOKENS + 4,
                ],
                [0, 1],
            ),
        )
        self.assertEqual(
            vocab_processor.numberize(['the', '..']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS + 1,
                    constants.START_WORD_ID,
                    constants.UNK_ID,
                ],
                [0, 1],
            ),
        )

        self.assertEqual(
            vocab_processor.denumberize([
                constants.MAX_SPECIAL_TOKENS + 1,
                constants.START_WORD_ID,
                constants.MAX_SPECIAL_TOKENS + 7,
                constants.MAX_SPECIAL_TOKENS + 4,
            ]),
            (['the', 'ine'], [0, 1])
        )
        self.assertEqual(
            vocab_processor.denumberize([
                constants.MAX_SPECIAL_TOKENS + 1,
                constants.START_WORD_ID,
                constants.MAX_SPECIAL_TOKENS + 5,
                constants.MAX_SPECIAL_TOKENS + 7,
                constants.MAX_SPECIAL_TOKENS + 1,
            ]),
            (['the', 'thin', 'the'], [0, 1, 4])
        )
        # when sequence does not have START_WORD_TOKEN
        self.assertEqual(
            vocab_processor.denumberize([
                constants.MAX_SPECIAL_TOKENS + 1,
                constants.MAX_SPECIAL_TOKENS + 5,
                constants.MAX_SPECIAL_TOKENS + 7,
                constants.MAX_SPECIAL_TOKENS + 1,
            ]),
            (['the', 'thin', 'the'], [0, 1, 3])
        )
        self.assertEqual(
            vocab_processor.get_token(constants.MAX_SPECIAL_TOKENS + 8),
            '\uffe4\u5c8d',
        )
        self.assertEqual(
            vocab_processor.get_token(constants.UNK_ID),
            constants.UNK_TOKEN,
        )
        self.assertEqual(
            vocab_processor.get_token(constants.MAX_SPECIAL_TOKENS + 100),
            None,
        )

    def test_dump_all_tokens(self):
        shortlist = ['.', 'the', ',', 'of']
        ngrams = ['e', 'th', '.', 'in', '\uffe4\u5c8d', '\uffe4\u9593']
        vocab_processor = VocabProcessor(serialize_vocab_thrift_obj_to_file(
            ttypes.BaseVocab(
                char_ngram_vocab=ttypes.CharNgramVocab(
                    words=self.generate_word_struct(shortlist),
                    ngrams=self.generate_word_struct(ngrams),
                    ngram_size=2,
                    generate_copy_tokens=False,
                )
            )
        ))
        output_file = tempfile.NamedTemporaryFile('r', delete=False)
        vocab_processor.dump_all_tokens(output_file.name)
        self.assertEqual(
            [line.strip() for line in output_file.readlines()],
            constants.SPECIAL_TOKENS + shortlist + ngrams,
        )
