from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile

from libfb.py import testutil
from language_technology.neural_mt.fbtranslate.vocab_processor import (
    VocabProcessor,
)
from language_technology.neural_mt.fbtranslate.vocab.vocab_writer import (
    save_vocab_to_file,
)
import language_technology.neural_mt.fbtranslate.vocab.\
    vocab_constants as constants


class WordVocabProcessTest(testutil.BaseFacebookTestCase):

    def test_simple(self):
        vocab_processor = VocabProcessor(save_vocab_to_file(
            vocab_type=constants.WORD_VOCAB_TYPE,
            tokens_and_counts=[('a', 1), ('b', 1)],
        ))
        self.assertEqual(
            vocab_processor.size(),
            constants.MAX_SPECIAL_TOKENS + 2,
        )
        self.assertEqual(
            vocab_processor.numberize(['a', 'b', 'c']),
            (
                [
                    constants.MAX_SPECIAL_TOKENS,
                    constants.MAX_SPECIAL_TOKENS + 1,
                    constants.UNK_ID
                ],
                [0, 1, 2],
            )
        )
        self.assertEqual(
            vocab_processor.denumberize([
                constants.MAX_SPECIAL_TOKENS,
                constants.MAX_SPECIAL_TOKENS + 1,
                constants.UNK_ID
            ]),
            (['a', 'b', constants.UNK_TOKEN], [0, 1, 2])
        )
        self.assertEqual(
            vocab_processor.denumberize([
                constants.MAX_SPECIAL_TOKENS,
                constants.MAX_SPECIAL_TOKENS - 1,
                constants.UNK_ID
            ]),
            (['a', constants.UNDEFINED_TOKEN, constants.UNK_TOKEN], [0, 1, 2])
        )
        self.assertEqual(
            vocab_processor.get_id('a'),
            constants.MAX_SPECIAL_TOKENS,
        )
        self.assertEqual(
            vocab_processor.get_id(constants.UNK_TOKEN),
            constants.UNK_ID,
        )
        self.assertEqual(
            vocab_processor.get_id('e'),
            None,
        )
        self.assertEqual(
            vocab_processor.get_token(constants.MAX_SPECIAL_TOKENS),
            'a',
        )
        self.assertEqual(
            vocab_processor.get_token(constants.UNK_ID),
            constants.UNK_TOKEN,
        )
        self.assertEqual(
            vocab_processor.get_token(constants.MAX_SPECIAL_TOKENS + 100),
            None,
        )
        self.assertEqual(
            vocab_processor.get_full_token(constants.MAX_SPECIAL_TOKENS),
            ('a', 1),
        )
        self.assertEqual(
            vocab_processor.get_full_token(constants.UNK_ID),
            (constants.UNK_TOKEN, 0),
        )
        self.assertEqual(
            vocab_processor.get_token(constants.MAX_SPECIAL_TOKENS + 100),
            None,
        )

    def test_numberize_file(self):
        vocab_processor = VocabProcessor(save_vocab_to_file(
            vocab_type=constants.WORD_VOCAB_TYPE,
            tokens_and_counts=[('a', 1), ('b', 1)],
        ))

        input_file = tempfile.NamedTemporaryFile('w', delete=False)
        input_file.write('a b c\nc b a\n')
        input_file.close()

        output_file = tempfile.NamedTemporaryFile('r', delete=False)
        vocab_processor.numberize_file(
            input_file.name,
            output_file.name,
            num_threads=10,
        )
        self.assertEqual(
            output_file.readlines(),
            [
                '{} {} {}\n'.format(
                    constants.MAX_SPECIAL_TOKENS,
                    constants.MAX_SPECIAL_TOKENS + 1,
                    constants.UNK_ID,
                ),
                '{} {} {}\n'.format(
                    constants.UNK_ID,
                    constants.MAX_SPECIAL_TOKENS + 1,
                    constants.MAX_SPECIAL_TOKENS,
                ),
            ],
        )
