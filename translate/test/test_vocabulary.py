#!/usr/bin/env python3

import os
import tempfile
import unittest

from collections import OrderedDict
from pytorch_translate.vocabulary import Vocabulary


class VocabularyTest(unittest.TestCase):
    def _text1_setup(self, path):
        with open(path, 'w') as f:
            f.write(
                r"""
                I am
                I am a sentence

                I am also sentence
                """
            )

    def _text1_expected_ordered_dict(self):
        return OrderedDict([
            ('<PAD>', -1),
            ('<GO>', -1),
            ('<EOS>', -1),
            ('<UNK>', -1),
            ('am', 3),
            ('I', 3),
            ('sentence', 2),
            ('also', 1),
            ('a', 1),
        ])

    def _text2_setup(self, path):
        with open(path, 'w') as f:
            f.write(
                r"""
                ce n' est pas une phrase .
                une phrase est
                une phrase
                """
            )

    def _text2_expected_ordered_dict(self):
        return OrderedDict([
            ('<PAD>', -1),
            ('<GO>', -1),
            ('<EOS>', -1),
            ('<UNK>', -1),
            ('une', 3),
            ('phrase', 3),
            ('est', 2),
            ('pas', 1),
            ('n\'', 1),
            ('ce', 1),
            ('.', 1),
        ])

    def _check_equal(self, vocab, expected_ordered_dict, top_n=None):
        size = len(expected_ordered_dict)
        if top_n is not None:
            size = min(top_n + Vocabulary.NUM_SPECIAL_TOKENS, size)
        self.assertEqual(size, vocab.vocab_size())

        id = 0
        for token, freq in expected_ordered_dict.items():
            if id >= size:
                break
            self.assertEqual(token, vocab.id_to_token(id))
            self.assertEqual(freq, vocab.id_to_freq(id))
            self.assertEqual(id, vocab.token_to_id(token))
            self.assertEqual(freq, vocab.token_to_freq(token))
            id += 1

    def test_load(self):
        # Tests loading from text file
        vocab = Vocabulary()
        fd, path = tempfile.mkstemp(text=True)
        self._text1_setup(path)
        vocab.load_tokenized_text(path)
        self._check_equal(vocab, self._text1_expected_ordered_dict())
        self.assertEqual(
            Vocabulary.UNK_ID, vocab.token_to_id('NONEXISTENT_TOKEN')
        )
        os.close(fd)
        os.remove(path)

        # Tests loading from serialized format
        vocab_serialized = Vocabulary()
        fd_serialized, path_serialized = tempfile.mkstemp(text=False)
        vocab.save(path_serialized)
        vocab_serialized.load_serialized(path_serialized)
        self._check_equal(vocab_serialized, self._text1_expected_ordered_dict())
        self.assertEqual(
            Vocabulary.UNK_ID, vocab.token_to_id('NONEXISTENT_TOKEN')
        )
        os.close(fd_serialized)
        os.remove(path_serialized)

    def test_load_top_n(self):
        TOP_N = 3
        vocab = Vocabulary()
        fd, path = tempfile.mkstemp(text=True)
        self._text2_setup(path)
        vocab.load_tokenized_text(path, top_n=TOP_N)
        self._check_equal(
            vocab, self._text2_expected_ordered_dict(), top_n=TOP_N
        )
        os.close(fd)
        os.remove(path)

    def test_repeated_load(self):
        # Tests re-using the same Vocabulary object to load different datasets.
        vocab = Vocabulary()

        fd_1, path_1 = tempfile.mkstemp(text=True)
        fd_2, path_2 = tempfile.mkstemp(text=True)
        fd_serialized, path_serialized = tempfile.mkstemp(text=False)

        # 1) Loads from text file 1 and prepare serialized file.
        self._text1_setup(path_1)
        vocab.load_tokenized_text(path_1)
        self._check_equal(vocab, self._text1_expected_ordered_dict())
        vocab.save(path_serialized)

        # 2) Loads from text file 2.
        self._text2_setup(path_2)
        vocab.load_tokenized_text(path_2)
        self._check_equal(vocab, self._text2_expected_ordered_dict())

        # 3) Loads from serialized version of file 1.
        vocab.load_serialized(path_serialized)
        self._check_equal(vocab, self._text1_expected_ordered_dict())

        # 4) Loads from text file 2 again.
        vocab.load_tokenized_text(path_2)
        self._check_equal(vocab, self._text2_expected_ordered_dict())

        os.close(fd_1)
        os.close(fd_2)
        os.close(fd_serialized)

        os.remove(path_1)
        os.remove(path_2)
        os.remove(path_serialized)

    def test_errors(self):
        vocab = Vocabulary()
        # Trying to use vocab before initializing should raise RuntimeError.
        self.assertRaises(RuntimeError, vocab.id_to_token, 0)
        self.assertRaises(RuntimeError, vocab.id_to_freq, 0)
        self.assertRaises(RuntimeError, vocab.token_to_id, 'foo')
        self.assertRaises(RuntimeError, vocab.token_to_freq, 'foo')
