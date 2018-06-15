#!/usr/bin/env python3

import unittest
import os

from pytorch_translate import data
from pytorch_translate import dictionary
from pytorch_translate.test import utils as test_utils


class TestInMemoryNumpyDataset(unittest.TestCase):
    def setUp(self):
        self.src_txt, self.trg_txt = test_utils.create_test_text_files()
        self.vocab_file_path = test_utils.make_temp_file()
        self.d = dictionary.Dictionary.build_vocab_file(
            corpus_files=[self.src_txt, self.trg_txt],
            vocab_file=self.vocab_file_path,
            max_vocab_size=0,
        )
        # src_ref is reversed, +1 for lua
        self.src_ref = [
            [107, 105, 103, 101],
            [105, 105, 103, 103, 101, 101],
            [103, 103, 103, 103, 101, 101, 101, 101],
            [101, 101, 101, 101, 101, 101, 101, 101, 101, 101],
        ]
        self.trg_ref = [
            [102, 102, 102, 102, 102, 102, 102, 102, 102, 102],
            [102, 102, 102, 102, 104, 104, 104, 104],
            [102, 102, 104, 104, 106, 106],
            [102, 104, 106, 108],
        ]
        self.lua_eos = self.d.eos_index + 1
        self.num_sentences = 4

    def tearDown(self):
        os.remove(self.src_txt)
        os.remove(self.trg_txt)
        os.remove(self.vocab_file_path)

    def test_parse(self):
        src_dataset = data.InMemoryNumpyDataset()
        trg_dataset = data.InMemoryNumpyDataset()
        for _ in range(2):
            src_dataset.parse(
                self.src_txt, self.d, reverse_order=True, append_eos=False
            )
            trg_dataset.parse(
                self.trg_txt, self.d, reverse_order=False, append_eos=True
            )
            self.assertEqual(self.num_sentences, len(src_dataset))
            self.assertEqual(self.num_sentences, len(trg_dataset))
            for i in range(self.num_sentences):
                self.assertListEqual(self.src_ref[i], src_dataset[i].tolist())
                self.assertListEqual(
                    self.trg_ref[i] + [self.lua_eos], trg_dataset[i].tolist()
                )

    def test_parse_oversampling(self):
        dataset = data.InMemoryNumpyDataset()
        factors = [(1, 0), (3, 2), (4, 4)]
        for o1, o2 in factors:
            corpora = [
                data.MultilingualCorpusConfig(
                    dialect_id=None,
                    data_file=self.trg_txt,
                    dict=self.d,
                    oversampling=o1,
                ),
                data.MultilingualCorpusConfig(
                    dialect_id=None,
                    data_file=self.trg_txt,
                    dict=self.d,
                    oversampling=o2,
                ),
            ]
            dataset.parse_multilingual(corpora)
            self.assertEqual((o1 + o2) * self.num_sentences, len(dataset))

    def test_parse_multiling(self):
        prepend_dataset = data.InMemoryNumpyDataset()
        append_dataset = data.InMemoryNumpyDataset()
        corpora = [
            data.MultilingualCorpusConfig(
                dialect_id=10, data_file=self.trg_txt, dict=self.d, oversampling=1
            ),
            data.MultilingualCorpusConfig(
                dialect_id=11, data_file=self.trg_txt, dict=self.d, oversampling=1
            ),
        ]
        lang1 = corpora[0].dialect_id + 1  # +1 for lua
        lang2 = corpora[1].dialect_id + 1  # +1 for lua
        prepend_dataset.parse_multilingual(
            corpora, reverse_order=False, append_eos=False, prepend_language_id=True
        )
        append_dataset.parse_multilingual(
            corpora, reverse_order=False, append_eos=False, prepend_language_id=False
        )
        self.assertEqual(2 * self.num_sentences, len(prepend_dataset))
        self.assertEqual(2 * self.num_sentences, len(append_dataset))
        for i in range(self.num_sentences):
            self.assertListEqual([lang1] + self.trg_ref[i], prepend_dataset[i].tolist())
            self.assertListEqual(self.trg_ref[i] + [lang1], append_dataset[i].tolist())
            self.assertListEqual(
                [lang2] + self.trg_ref[i],
                prepend_dataset[i + self.num_sentences].tolist(),
            )
            self.assertListEqual(
                self.trg_ref[i] + [lang2],
                append_dataset[i + self.num_sentences].tolist(),
            )
