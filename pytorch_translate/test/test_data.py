#!/usr/bin/env python3

import os
import tempfile
import unittest

from fairseq.data import LanguagePairDataset
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset
from pytorch_translate import dictionary, preprocess
from pytorch_translate.data import data
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestLoadData(unittest.TestCase):
    def test_load_data_single_path(self):
        test_args = test_utils.ModelParamsDict()
        test_args.source_lang = "en"
        test_args.target_lang = "fr"
        test_args.log_verbose = False
        src_dict, tgt_dict = test_utils.create_vocab_dictionaries()
        src_text_file, tgt_text_file = test_utils.create_test_text_files()
        src_bin_path = preprocess.binarize_text_file(
            text_file=src_text_file,
            dictionary=src_dict,
            output_path=tempfile.NamedTemporaryFile().name,
            append_eos=True,
            reverse_order=False,
        )
        tgt_bin_path = preprocess.binarize_text_file(
            text_file=tgt_text_file,
            dictionary=tgt_dict,
            output_path=tempfile.NamedTemporaryFile().name,
            append_eos=True,
            reverse_order=False,
        )
        task = tasks.PytorchTranslateTask(test_args, src_dict, tgt_dict)
        split = "0"
        task.load_dataset(split, src_bin_path, tgt_bin_path)
        self.assertEqual(len(task.datasets[split]), 4)
        self.assertIsInstance(task.datasets[split], LanguagePairDataset)

    def test_load_data_multi_path(self):
        test_args = test_utils.ModelParamsDict()
        test_args.source_lang = "en"
        test_args.target_lang = "fr"
        test_args.log_verbose = False
        src_dict, tgt_dict = test_utils.create_vocab_dictionaries()
        num_paths = 4
        src_bin_path, tgt_bin_path = {}, {}
        for i in range(num_paths):
            src_text_file, tgt_text_file = test_utils.create_test_text_files()
            src_bin_path[i] = preprocess.binarize_text_file(
                text_file=src_text_file,
                dictionary=src_dict,
                output_path=tempfile.NamedTemporaryFile().name,
                append_eos=True,
                reverse_order=False,
            )
            tgt_bin_path[i] = preprocess.binarize_text_file(
                text_file=tgt_text_file,
                dictionary=tgt_dict,
                output_path=tempfile.NamedTemporaryFile().name,
                append_eos=True,
                reverse_order=False,
            )
        task = tasks.PytorchTranslateTask(test_args, src_dict, tgt_dict)
        split = "1"
        task.load_dataset(split, src_bin_path, tgt_bin_path)
        self.assertEqual(len(task.datasets[split]), 16)
        self.assertIsInstance(task.datasets[split], MultiCorpusSampledDataset)


class TestInMemoryNumpyDataset(unittest.TestCase):
    def setUp(self):
        self.src_txt, self.trg_txt = test_utils.create_test_text_files()
        self.vocab_file_path = test_utils.make_temp_file()
        self.d = dictionary.Dictionary.build_vocab_file(
            corpus_files=[self.src_txt, self.trg_txt],
            vocab_file=self.vocab_file_path,
            max_vocab_size=0,
            padding_factor=1,  # don't add extra padding symbols
        )
        # src_ref is reversed
        self.src_ref = [
            [106, 104, 102, 100],
            [104, 104, 102, 102, 100, 100],
            [102, 102, 102, 102, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        ]
        self.trg_ref = [
            [101, 101, 101, 101, 101, 101, 101, 101, 101, 101],
            [101, 101, 101, 101, 103, 103, 103, 103],
            [101, 101, 103, 103, 105, 105],
            [101, 103, 105, 107],
        ]
        self.src_txt_numberized, self.trg_txt_numberized = test_utils.create_test_numberized_data_files(
            self.src_ref, self.trg_ref, reverse_source=True
        )
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
                    self.trg_ref[i] + [self.d.eos_index], trg_dataset[i].tolist()
                )

    def test_parse_numberize(self):
        src_dataset = data.InMemoryNumpyDataset()
        trg_dataset = data.InMemoryNumpyDataset()
        for _ in range(2):
            src_dataset.parse(
                self.src_txt_numberized,
                self.d,
                reverse_order=True,
                append_eos=False,
                already_numberized=True,
            )
            trg_dataset.parse(
                self.trg_txt_numberized,
                self.d,
                reverse_order=False,
                append_eos=True,
                already_numberized=True,
            )
            self.assertEqual(self.num_sentences, len(src_dataset))
            self.assertEqual(self.num_sentences, len(trg_dataset))
            for i in range(self.num_sentences):
                self.assertListEqual(self.src_ref[i], src_dataset[i].tolist())
                self.assertListEqual(
                    self.trg_ref[i] + [self.d.eos_index], trg_dataset[i].tolist()
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
        lang1 = corpora[0].dialect_id
        lang2 = corpora[1].dialect_id
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
