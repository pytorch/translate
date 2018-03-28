from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import string

from libfb.py import testutil
from language_technology.neural_mt.fbtranslate.vocab_processor import (
    VocabProcessor,
)
from language_technology.neural_mt.fbtranslate.vocab.vocab_writer import (
    save_vocab_to_file,
)
import language_technology.neural_mt.fbtranslate.vocab.\
    vocab_constants as constants


class VocabTest(testutil.BaseFacebookTestCase):

    MAX_ITERATIONS = 1000
    MAX_LENGTH = 10

    def generate_tokens_struct(self, tokens):
        return [(token, 1) for token in tokens]

    def test_word_vocab_invariant(self):
        vocab_processor = VocabProcessor(save_vocab_to_file(
            vocab_type=constants.WORD_VOCAB_TYPE,
            tokens_and_counts=self.generate_tokens_struct(
                string.ascii_lowercase,
            ),
        ))

        for i in range(self.MAX_ITERATIONS):
            possible_token_ids = [
                index + constants.MAX_SPECIAL_TOKENS
                for index in range(len(string.ascii_lowercase))
            ] + list(constants.SPECIAL_TOKENS_TO_ID.values())
            expected_words = [
                random.choice(possible_token_ids)
                for i in range(random.randint(0, self.MAX_LENGTH))
            ]
            actual_words = vocab_processor.numberize(
                vocab_processor.denumberize(expected_words)[0]
            )[0]
            self.assertEqual(
                expected_words,
                actual_words,
                msg=(
                    'The invariant didn\'t hold:\n'
                    'denumberize({token_ids}) = {denum_token_ids},\n'
                    'numberize(denumberize([{token_ids}]) = {num_token_ids},\n'
                    'but it\'s not equal to original {token_ids}'.format(
                        token_ids=expected_words,
                        denum_token_ids=(
                            vocab_processor.denumberize(expected_words)[0]
                        ),
                        num_token_ids=actual_words,
                    )
                )
            )
