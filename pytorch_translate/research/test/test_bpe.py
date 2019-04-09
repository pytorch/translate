#!/usr/bin/env python3

import unittest
from collections import Counter
from unittest.mock import Mock, patch

from pytorch_translate.research.unsupervised_morphology import bpe


txt_content = ["123 124 234 345", "112 122 123 345", "123456789", "123456 456789"]


class TestBPE(unittest.TestCase):
    def test_vocab_init(self):
        bpe_model = bpe.BPE()

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            bpe_model.init_vocab(txt_path="no_exist_file.txt")

            vocab_items = Counter()
            for vocab_entry in bpe_model.vocab.keys():
                items = vocab_entry.split()
                for item in items:
                    vocab_items[item] += bpe_model.vocab[vocab_entry]

            assert vocab_items[bpe_model.eow_symbol] == 11
            assert vocab_items["3"] == 7
            assert len(vocab_items) == 10
            assert "12" not in vocab_items
            assert "123" not in vocab_items

    def test_best_candidate(self):
        bpe_model = bpe.BPE()

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            bpe_model.init_vocab(txt_path="no_exist_file.txt")

            assert bpe_model.get_best_candidate() == ("1", "2")

    def test_bpe_merge(self):
        bpe_model = bpe.BPE()

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            bpe_model.init_vocab(txt_path="no_exist_file.txt")

            # Trying merging a candidate that does not exist.
            vocab_size = bpe_model.merge_candidate_into_vocab(("3", "1"))
            assert vocab_size == 10

            # Trying merging a candidate that does exists.
            vocab_size = bpe_model.merge_candidate_into_vocab(("2", "3"))
            assert vocab_size == 11

            # Trying merging a candidate that does exists. Entry "3" should remove
            # from vocab.
            vocab_size = bpe_model.merge_candidate_into_vocab(("3", "4"))
            assert vocab_size == 11

            # Trying merging a candidate that does not exist.
            vocab_size = bpe_model.merge_candidate_into_vocab(
                ("3", bpe_model.eow_symbol)
            )
            assert vocab_size == 11

    def test_merge_pattern(self):
        pattern1 = bpe.BPE.get_merge_pattern("c c")
        assert pattern1.sub("cc", "x|x?^d@@ c c ^d") == "x|x?^d@@ cc ^d"

        pattern2 = bpe.BPE.get_merge_pattern("^d@ @c")
        assert (
            pattern2.sub("^d@@c", "^d@ @cx|x? ^d@ @c c ^d") == "^d@ @cx|x? ^d@@c c ^d"
        )

        pattern3 = bpe.BPE.get_merge_pattern("x| x")
        assert (
            pattern3.sub("x|x", "^d@ @c x| x ?^d@ @c c ^d") == "^d@ @c x|x ?^d@ @c c ^d"
        )

    def test_build_vocab(self):
        bpe_model = bpe.BPE()

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))

            # Trying to build a vocab more than the possible size
            vocab_size = bpe_model.build_vocab(
                txt_path="no_exist_file.txt", vocab_size=200
            )
            # Asserting that we go back to the original size (number of word types.)
            assert vocab_size == 9
            assert bpe_model.max_bpe_len == 9 + len(bpe_model.eow_symbol)

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            # Trying to build a vocab with an acceptable size.
            vocab_size = bpe_model.build_vocab(
                txt_path="no_exist_file.txt", vocab_size=12
            )
            # asserting that the size is as expected.
            assert vocab_size == 12
            assert bpe_model.max_bpe_len == len(bpe_model.eow_symbol)
