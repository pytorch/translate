from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import tempfile

from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map

import language_technology.neural_mt.fbtranslate.vocab.\
    vocab_constants as constants

cdef extern from \
'language_technology/neural_mt/fbtranslate/vocab/BaseVocabProcessor.h' nogil:
    cdef cppclass TokenAndCount \
    'facebook::language_technology::neural_mt::TokenAndCount':
        string token
        int count
    cdef cppclass BaseVocabProcessor \
    'facebook::language_technology::neural_mt::BaseVocabProcessor':
        pair[vector[int], vector[int]] numberize(const vector[string] &)
        pair[vector[string], vector[int]] tokenize(const vector[string] &)
        pair[vector[string], vector[int]] denumberize(const vector[int] &)
        void numberizeFile(const string&, const string&, int) except +
        void tokenizeFile(const string&, const string&, int) except +
        int getId(const string&)
        pair[bool, string] getToken(int)
        pair[bool, pair[string, int]] getFullToken(int)
        int size()
        vector[int] getProfanityTokenIds()
        void setProfanityTokenIds(const vector[string] &)

cdef extern from \
'language_technology/neural_mt/fbtranslate/vocab/Loader.h' nogil:
    shared_ptr[BaseVocabProcessor] loadVocabProcessorFromFile \
    'facebook::language_technology::neural_mt::loadVocabProcessorFromFile' \
    (const string &) except +


cdef class VocabProcessor:
    cdef shared_ptr[BaseVocabProcessor] vocab_processor
    cdef public unicode vocab_file

    def __cinit__(self, vocab_file):
        self.vocab_file = vocab_file
        self.vocab_processor = loadVocabProcessorFromFile(
            vocab_file.encode('utf-8')
        )

    def size(self):
        return self.vocab_processor.get().size()

    def numberize(self, tokens):
        res = self.vocab_processor.get().numberize(
            [token.encode('utf-8') for token in tokens]
        )
        return res.first, res.second

    def tokenize(self, tokens):
        res = self.vocab_processor.get().tokenize(
            [token.encode('utf-8') for token in tokens]
        )
        return res.first, res.second

    def get_id(self, token):
        token_id = self.vocab_processor.get().getId(token.encode('utf-8'))
        return token_id if token_id != constants.INVALID_ID else None

    def get_token(self, token_id):
        token = self.vocab_processor.get().getToken(token_id)
        return token.second.decode('utf-8') if token.first else None

    def get_full_token(self, token_id):
        token = self.vocab_processor.get().getFullToken(token_id)
        return (
            token.second.first.decode('utf-8'),
            token.second.second,
        ) if token.first else None

    def numberize_file(self, input_path, output_path, num_threads):
        self.vocab_processor.get().numberizeFile(
            input_path.encode('utf-8'),
            output_path.encode('utf-8'),
            num_threads,
        )

    def tokenize_file(self, input_path, output_path, num_threads):
        self.vocab_processor.get().tokenizeFile(
            input_path.encode('utf-8'),
            output_path.encode('utf-8'),
            num_threads,
        )

    def denumberize(self, const vector[int] & token_ids):
        res = self.vocab_processor.get().denumberize(token_ids)
        return [token.decode('utf-8') for token in res.first], res.second

    def dump_all_tokens(self, output_path):
        size = self.size()
        with codecs.open(output_path, 'w', 'utf-8') as output_file:
            for token in constants.SPECIAL_TOKENS:
                output_file.write(token + '\n')
            for token_id in range(constants.MAX_SPECIAL_TOKENS, size):
                token = self.get_token(token_id)
                assert token is not None
                output_file.write(token + '\n')

    # TODO: do profanity token mapping outside of VocabProcessor
    def set_profanity_ids(self, tokens):
        self.vocab_processor.get().setProfanityTokenIds(tokens)

    def get_profanity_ids(self):
        return self.vocab_processor.get().getProfanityTokenIds()
