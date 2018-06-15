#!/usr/bin/env python3

import unittest
import os

from pytorch_translate import dictionary
from pytorch_translate.test import utils as test_utils


class TestDictionary(unittest.TestCase):
    def test_base(self):
        d = dictionary.Dictionary()
        self.assertEqual(len(d), d.nspecial)

    def test_build_vocab_file(self):
        src_txt, trg_txt = test_utils.create_test_text_files()
        tmp_prefix = test_utils.make_temp_file()
        src_dict1 = dictionary.Dictionary.build_vocab_file(
            corpus_files=[src_txt], vocab_file=f"{tmp_prefix}.src1", max_vocab_size=1000
        )
        src_dict2 = dictionary.Dictionary.build_vocab_file(
            corpus_files=[src_txt, src_txt, src_txt],
            vocab_file=f"{tmp_prefix}.src2",
            max_vocab_size=1000,
        )
        trg_dict1 = dictionary.Dictionary.build_vocab_file(
            corpus_files=[trg_txt], vocab_file=f"{tmp_prefix}.trg1", max_vocab_size=1000
        )
        trg_dict2 = dictionary.Dictionary.build_vocab_file(
            corpus_files=[trg_txt, trg_txt, trg_txt],
            vocab_file=f"{tmp_prefix}.trg2",
            max_vocab_size=1000,
        )
        srctrg_dict = dictionary.Dictionary.build_vocab_file(
            corpus_files=[src_txt, trg_txt],
            vocab_file=f"{tmp_prefix}.srctrg",
            max_vocab_size=1000,
        )
        nspecial = src_dict1.nspecial
        self.assertEqual(len(src_dict1), nspecial + 4)
        self.assertEqual(len(trg_dict1), nspecial + 4)
        self.assertEqual(len(srctrg_dict), nspecial + 8)
        for s in src_dict1.symbols:
            self.assertIn(s, srctrg_dict.symbols)
        for s in trg_dict1.symbols:
            self.assertIn(s, srctrg_dict.symbols)
        src_dict1_loaded = dictionary.Dictionary.load(f"{tmp_prefix}.src1")
        src_dict2_loaded = dictionary.Dictionary.load(f"{tmp_prefix}.src2")
        trg_dict1_loaded = dictionary.Dictionary.load(f"{tmp_prefix}.trg1")
        trg_dict2_loaded = dictionary.Dictionary.load(f"{tmp_prefix}.trg2")
        self._assert_vocab_equal(src_dict1, src_dict2)
        self._assert_vocab_equal(src_dict1, src_dict1_loaded)
        self._assert_vocab_equal(src_dict1, src_dict2_loaded)
        self._assert_vocab_equal(trg_dict1, trg_dict2)
        self._assert_vocab_equal(trg_dict1, trg_dict1_loaded)
        self._assert_vocab_equal(trg_dict1, trg_dict2_loaded)
        for c in range(nspecial, nspecial + 4):
            self.assertEqual(src_dict1.count[c], src_dict1_loaded.count[c])
            self.assertEqual(src_dict2.count[c], src_dict2_loaded.count[c])
            self.assertEqual(src_dict1.count[c] * 3, src_dict2.count[c])
            self.assertEqual(trg_dict1.count[c], trg_dict1_loaded.count[c])
            self.assertEqual(trg_dict2.count[c], trg_dict2_loaded.count[c])
            self.assertEqual(trg_dict1.count[c] * 3, trg_dict2.count[c])
        os.remove(f"{tmp_prefix}.src1")
        os.remove(f"{tmp_prefix}.src2")
        os.remove(f"{tmp_prefix}.trg1")
        os.remove(f"{tmp_prefix}.trg2")
        os.remove(src_txt)
        os.remove(trg_txt)

    def test_build_vocab_file_max_vocab(self):
        src_txt, trg_txt = test_utils.create_test_text_files()
        tmp_prefix = test_utils.make_temp_file()
        src_dict1 = dictionary.Dictionary.build_vocab_file(
            corpus_files=[src_txt], vocab_file=f"{tmp_prefix}.src1", max_vocab_size=1
        )
        src_dict2 = dictionary.Dictionary.build_vocab_file(
            corpus_files=[src_txt], vocab_file=f"{tmp_prefix}.src2", max_vocab_size=2
        )
        src_dict3 = dictionary.Dictionary.build_vocab_file(
            corpus_files=[src_txt], vocab_file=f"{tmp_prefix}.src3", max_vocab_size=104
        )
        src_dict4 = dictionary.Dictionary.build_vocab_file(
            corpus_files=[src_txt], vocab_file=f"{tmp_prefix}.src4", max_vocab_size=0
        )
        self.assertEqual(src_dict1.nspecial + 1, len(src_dict1))
        self.assertEqual(src_dict2.nspecial + 2, len(src_dict2))
        self.assertEqual(src_dict3.nspecial + 4, len(src_dict3))
        self._assert_vocab_equal(src_dict3, src_dict4)
        os.remove(f"{tmp_prefix}.src1")
        os.remove(f"{tmp_prefix}.src2")
        os.remove(f"{tmp_prefix}.src3")
        os.remove(f"{tmp_prefix}.src4")
        os.remove(src_txt)
        os.remove(trg_txt)

    def _assert_vocab_equal(self, d1, d2):
        self.assertDictEqual(d1.indices, d2.indices)
        self.assertSetEqual(d1.lexicon_indices, d2.lexicon_indices)
        self.assertListEqual(d1.symbols, d2.symbols)


class TestMaxVocabDictionary(unittest.TestCase):
    def test_push(self):
        max_vocab_dict = dictionary.MaxVocabDictionary()
        src_txt, trg_txt = test_utils.create_test_text_files()
        tmp_prefix = test_utils.make_temp_file()
        src_dict = dictionary.Dictionary.build_vocab_file(
            corpus_files=[src_txt], vocab_file=f"{tmp_prefix}.src", max_vocab_size=1000
        )
        srctrg_dict = dictionary.Dictionary.build_vocab_file(
            corpus_files=[src_txt, trg_txt],
            vocab_file=f"{tmp_prefix}.srctrg",
            max_vocab_size=1000,
        )
        self.assertEqual(len(max_vocab_dict), max_vocab_dict.nspecial)
        max_vocab_dict.push(src_dict)
        self.assertEqual(len(max_vocab_dict), len(src_dict))
        max_vocab_dict.push(srctrg_dict)
        self.assertEqual(len(max_vocab_dict), len(srctrg_dict))
        max_vocab_dict.push(src_dict)
        self.assertEqual(len(max_vocab_dict), len(srctrg_dict))
        os.remove(f"{tmp_prefix}.src")
        os.remove(f"{tmp_prefix}.srctrg")
        os.remove(src_txt)
        os.remove(trg_txt)
