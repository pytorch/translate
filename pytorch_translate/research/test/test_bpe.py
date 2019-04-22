#!/usr/bin/env python3

import shutil
import tempfile
import unittest
from collections import Counter
from multiprocessing import Pool
from os import path
from unittest.mock import Mock, patch

from pytorch_translate.research.test import morphology_test_utils as morph_utils
from pytorch_translate.research.unsupervised_morphology import bilingual_bpe, bpe


txt_content = ["123 124 234 345", "112 122 123 345", "123456789", "123456 456789"]


class TestBPE(unittest.TestCase):
    def test_vocab_init(self):
        bpe_model = bpe.BPE()

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            bpe_model._init_vocab(txt_path="no_exist_file.txt")

            vocab_items = Counter()
            for (vocab_entry, freq) in bpe_model.current_train_data:
                for item in vocab_entry:
                    vocab_items[item] += freq

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
            bpe_model._init_vocab(txt_path="no_exist_file.txt")
            num_cpus = 3
            pool = Pool(processes=num_cpus)
            assert bpe_model.get_best_candidate(num_cpus=num_cpus, pool=pool) == (
                "1",
                "2",
            )

    def test_bpe_merge(self):
        bpe_model = bpe.BPE()

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            bpe_model._init_vocab(txt_path="no_exist_file.txt")
            num_cpus = 3
            pool = Pool(processes=num_cpus)

            # Trying merging a candidate that does not exist.
            vocab_size = bpe_model.merge_candidate_into_vocab(
                candidate=("3", "1"), num_cpus=num_cpus, pool=pool
            )
            assert vocab_size == 10

            # Trying merging a candidate that does exists.
            vocab_size = bpe_model.merge_candidate_into_vocab(
                candidate=("2", "3"), num_cpus=num_cpus, pool=pool
            )
            assert vocab_size == 11

            # Trying merging a candidate that does exists. Entry "3" should remove
            # from vocab.
            vocab_size = bpe_model.merge_candidate_into_vocab(
                candidate=("3", "4"), num_cpus=num_cpus, pool=pool
            )
            assert vocab_size == 11

            # Trying merging a candidate that does not exist.
            vocab_size = bpe_model.merge_candidate_into_vocab(
                candidate=("3", bpe_model.eow_symbol), num_cpus=num_cpus, pool=pool
            )
            assert vocab_size == 11

    def test_build_vocab(self):
        bpe_model = bpe.BPE()

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))

            # Trying to build a vocab more than the possible size
            vocab_size = bpe_model.build_vocab(
                txt_path="no_exist_file.txt", vocab_size=20, num_cpus=3
            )
            # Asserting that we go back to the original size (number of word types.)
            assert vocab_size == 9
            assert bpe_model.max_bpe_len == 9 + len(bpe_model.eow_symbol)

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            # Trying to build a vocab with an acceptable size.
            vocab_size = bpe_model.build_vocab(
                txt_path="no_exist_file.txt", vocab_size=12, num_cpus=3
            )
            # asserting that the size is as expected.
            assert vocab_size == len(bpe_model.vocab) == 12
            assert bpe_model.max_bpe_len == len(bpe_model.eow_symbol)

    def test_segment_word(self):
        bpe_model = bpe.BPE()

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))

            bpe_model.build_vocab(
                txt_path="no_exist_file.txt", vocab_size=12, num_cpus=3
            )
            assert bpe_model.segment_word("1234") == ["12", "34", bpe_model.eow_symbol]

            # Giving unknown character sequence
            assert bpe_model.segment_word("12634") == [
                "12",
                "6",
                "34",
                bpe_model.eow_symbol,
            ]

    def test_segment_file(self):
        bpe_model = bpe.BPE()

        tmp_dir = tempfile.mkdtemp()
        input_file, output_file = (
            path.join(tmp_dir, "test.in"),
            path.join(tmp_dir, "test1.out"),
        )

        with open(input_file, "w", encoding="utf-8") as writer:
            writer.write("\n".join(txt_content))
        bpe_model.build_vocab(txt_path=input_file, vocab_size=12, num_cpus=3)

        output = []
        for line in txt_content:
            cur_line_output = []
            for word in line.strip().split():
                cur_line_output.append(" ".join(bpe_model.segment_word(word)))
            output.append(" ".join(cur_line_output))
            output.append("\n")
        expected_output = "".join(output).strip()

        bpe_model.segment_txt(input_path=input_file, output_path=output_file)
        model_output = open(output_file, "r", encoding="utf-8").read().strip()
        assert expected_output == model_output

        shutil.rmtree(tmp_dir)

    def test_bilingual_bpe_init(self):
        """
            This looks more like an integration test because each subpeace is tested
            in different places.
        """
        bpe_model = bilingual_bpe.BilingualBPE()
        tmp_dir, f1, f2 = morph_utils.get_two_different_tmp_files()
        bpe_model._init_params(
            src_txt_path=f1, dst_txt_path=f2, num_ibm_iters=3, num_cpus=3
        )
        assert len(bpe_model.bpe_probs_from_alignment) == 80
        assert bpe_model.eow_symbol in bpe_model.bpe_probs_from_alignment

        shutil.rmtree(tmp_dir)

    def test_calc_word_probs(self):
        with patch("builtins.open") as mock_open:
            bpe_model = bilingual_bpe.BilingualBPE()
            mock_open.return_value.__enter__ = mock_open
            mock_open.return_value.__iter__ = Mock(return_value=iter(txt_content))
            v = bpe_model._calc_word_probs(txt_path="no_exist_file.txt")
            assert len(v) == 9
            assert v["123"] == 2 / 11
            assert v["123456789"] == 1 / 11

    def test_best_candidate_bilingual(self):
        bpe_model = bilingual_bpe.BilingualBPE()
        tmp_dir, f1, f2 = morph_utils.get_two_different_tmp_files()
        num_cpus = 3
        pool = Pool(num_cpus)
        bpe_model._init_params(
            src_txt_path=f1, dst_txt_path=f2, num_ibm_iters=3, num_cpus=num_cpus
        )

        b1 = bpe_model.get_best_candidate(num_cpus=num_cpus, pool=pool)
        c1 = bpe_model.get_best_candidate(num_cpus=num_cpus, pool=pool)
        # For the best step, it is the same as monolingual.
        assert b1 == c1

        shutil.rmtree(tmp_dir)

    def test_build_bilingual_vocab(self):
        bpe_model = bilingual_bpe.BilingualBPE()
        tmp_dir, f1, f2 = morph_utils.get_two_different_tmp_files()
        vocab_size = bpe_model.build_vocab(
            src_txt_path=f1, dst_txt_path=f2, vocab_size=12, num_ibm_iters=3, num_cpus=3
        )
        assert vocab_size == len(bpe_model.vocab) == 12
        shutil.rmtree(tmp_dir)
