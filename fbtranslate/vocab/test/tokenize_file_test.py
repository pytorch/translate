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


class TokenizeFileTest(testutil.BaseFacebookTestCase):

    def generate_tokens_struct(self, tokens):
        return [(token, 1) for token in tokens]

    def test_word_vocab_tokenize_file(self):
        vocab_processor = VocabProcessor(save_vocab_to_file(
            vocab_type=constants.WORD_VOCAB_TYPE,
            tokens_and_counts=self.generate_tokens_struct(['a', 'b']),
        ))

        input_file = tempfile.NamedTemporaryFile('w', delete=False)
        input_file.write('a b c\nc b a\n')
        input_file.close()

        output_file = tempfile.NamedTemporaryFile('r', delete=False)
        vocab_processor.tokenize_file(
            input_file.name,
            output_file.name,
            num_threads=10,
        )
        self.assertEqual(
            output_file.readlines(),
            [
                'a b {}\n'.format(constants.UNK_TOKEN),
                '{} b a\n'.format(constants.UNK_TOKEN),
            ],
        )
