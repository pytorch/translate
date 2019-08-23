#!/usr/bin/env python3

import os
import tempfile
import unittest

import numpy as np
import torch
from fairseq.data import LanguagePairDataset, NoisingDataset
from fairseq.data.concat_dataset import ConcatDataset
from fairseq.data.noising import UnsupervisedMTNoising
from fairseq_cli.preprocess import (
    binarize,
    dataset_dest_prefix,
    options as preprocess_options,
)
from pytorch_translate import preprocess
from pytorch_translate.data import char_data, data, dictionary
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestLoadData(unittest.TestCase):
    """
    This function tests loading from .npz data.
    """

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

    """
    This function tests loading from .idx + .bin data for fairseq compatibility.
    """

    def test_load_data_single_path_idx_bin(self):
        test_args = test_utils.ModelParamsDict()
        test_args.source_lang = "en"
        test_args.target_lang = "fr"
        test_args.log_verbose = False
        src_dict, tgt_dict = test_utils.create_vocab_dictionaries()
        src_text_file, tgt_text_file = test_utils.create_test_text_files()
        task = tasks.PytorchTranslateTask(test_args, src_dict, tgt_dict)
        with tempfile.TemporaryDirectory() as destdir:
            preprocess_args = [
                "--source-lang",
                test_args.source_lang,
                "--target-lang",
                test_args.target_lang,
                "--destdir",
                destdir,
            ]
            preproc_parser = preprocess_options.get_preprocessing_parser()
            preproc_args = preproc_parser.parse_args(preprocess_args)
            preproc_args.dataset_impl = "regular"  # No MMP
            split = "train"
            binarize(
                preproc_args,
                src_text_file,
                src_dict,
                split,
                test_args.source_lang,
                offset=0,
                end=-1,
            )
            binarize(
                preproc_args,
                tgt_text_file,
                tgt_dict,
                split,
                test_args.target_lang,
                offset=0,
                end=-1,
            )
            src_path = dataset_dest_prefix(preproc_args, split, test_args.source_lang)
            tgt_path = dataset_dest_prefix(preproc_args, split, test_args.target_lang)
            task.load_dataset(split, src_path, tgt_path, is_npz=False)
            self.assertEqual(len(task.datasets[split]), 4)
            self.assertIsInstance(task.datasets[split], LanguagePairDataset)

    def _prepare_data_multi_path(self, num_paths):
        test_args = test_utils.ModelParamsDict()
        test_args.source_lang = "en"
        test_args.target_lang = "fr"
        test_args.log_verbose = False
        test_args.dataset_upsampling = None
        test_args.dataset_relative_ratio = None
        src_dict, tgt_dict = test_utils.create_vocab_dictionaries()
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
        return test_args, src_dict, tgt_dict, src_bin_path, tgt_bin_path

    def test_load_data_multi_path(self):
        num_paths = 4
        test_args, src_dict, tgt_dict, src_bin_path, tgt_bin_path = self._prepare_data_multi_path(
            num_paths
        )
        task = tasks.PytorchTranslateTask(test_args, src_dict, tgt_dict)
        split = "1"
        task.load_dataset(split, src_bin_path, tgt_bin_path)
        self.assertEqual(len(task.datasets[split]), 16)
        self.assertIsInstance(task.datasets[split], ConcatDataset)

    def test_load_data_noising(self):
        num_paths = 4
        test_args, src_dict, tgt_dict, src_bin_path, tgt_bin_path = self._prepare_data_multi_path(
            num_paths
        )
        test_args.word_dropout_prob_map = str({"en-fr": {0: 0.1}})
        task = tasks.PytorchTranslateTask(test_args, src_dict, tgt_dict)
        split = "1"
        task.load_dataset(split, src_bin_path, tgt_bin_path)
        self.assertEqual(len(task.datasets[split]), 16)
        self.assertIsInstance(task.datasets[split].datasets[0].src, NoisingDataset)


class TestInMemoryIndexedDataset(unittest.TestCase):
    def setUp(self):
        self.src_txt, self.trg_txt = test_utils.create_test_text_files()
        self.vocab_file_path = test_utils.make_temp_file()
        self.d = dictionary.Dictionary.build_vocab_file(
            corpus_files=[self.src_txt, self.trg_txt],
            vocab_file=self.vocab_file_path,
            max_vocab_size=0,
            padding_factor=1,  # don't add extra padding symbols
        )
        self.char_dict = dictionary.Dictionary.build_vocab_file(
            corpus_files=[self.src_txt, self.trg_txt],
            vocab_file=self.vocab_file_path,
            max_vocab_size=0,
            is_char_vocab=True,
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
        src_dataset = data.InMemoryIndexedDataset()
        trg_dataset = data.InMemoryIndexedDataset()
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
        src_dataset = data.InMemoryIndexedDataset()
        trg_dataset = data.InMemoryIndexedDataset()
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
        dataset = data.InMemoryIndexedDataset()
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
        prepend_dataset = data.InMemoryIndexedDataset()
        append_dataset = data.InMemoryIndexedDataset()
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

    def test_subsample_pair_dataset(self):
        src_dataset = data.InMemoryIndexedDataset()
        trg_dataset = data.InMemoryIndexedDataset()
        for _ in range(5):
            src_dataset.parse(
                self.src_txt, self.d, reverse_order=True, append_eos=False
            )
            trg_dataset.parse(
                self.trg_txt, self.d, reverse_order=False, append_eos=True
            )

        pair_dataset = LanguagePairDataset(
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            src_dict=self.d,
            tgt=trg_dataset,
            tgt_sizes=trg_dataset.sizes,
            tgt_dict=self.d,
            left_pad_source=False,
        )

        data.subsample_pair_dataset(pair_dataset, 2)
        self.assertEqual(len(pair_dataset.src), 2)
        self.assertEqual(pair_dataset.src_sizes.size, 2)
        self.assertEqual(len(pair_dataset.tgt), 2)
        self.assertEqual(pair_dataset.tgt_sizes.size, 2)

    def test_subsample_dataset(self):
        """
        Test the InMemoryIndexedDataset.subsample() method, ensuring that the
        examples produced by the dataset are correctly permuted according to
        the indices argument.
        """
        trg_dataset = data.InMemoryIndexedDataset()

        trg_dataset.parse(self.trg_txt, self.d, reverse_order=False, append_eos=True)

        indices = np.random.permutation(len(trg_dataset))[:2]
        token_samples = [trg_dataset[i] for i in indices]
        trg_dataset.subsample(indices)
        for i in range(2):
            assert all(trg_dataset[i].numpy() == token_samples[i].numpy())

    def test_subsample_char_dataset(self):
        """
        Test the InMemoryNumpyWordCharDataset.subsample() method, ensuring that
        the examples produced by the dataset are correctly permuted according to
        the indices argument.
        """
        src_dataset = char_data.InMemoryNumpyWordCharDataset()
        src_dataset.parse(
            self.src_txt, self.d, self.char_dict, reverse_order=True, append_eos=False
        )

        indices = np.random.permutation(len(src_dataset))[:2]
        token_samples = [src_dataset.get_tokens(i) for i in indices]
        char_samples = [src_dataset.get_chars_list(i) for i in indices]
        src_dataset.subsample(indices)
        for i in range(2):
            assert all(src_dataset.get_tokens(i).numpy() == token_samples[i].numpy())
            orig_chars_list = char_samples[i]
            sampled_chars_list = src_dataset.get_chars_list(i)
            assert len(sampled_chars_list) == len(orig_chars_list)
            for sampled_chars, orig_chars in zip(sampled_chars_list, orig_chars_list):
                assert all(sampled_chars.numpy() == orig_chars.numpy())

    def test_collate_char_dataset(self):
        src_dataset = char_data.InMemoryNumpyWordCharDataset()
        src_dataset.parse(
            self.src_txt, self.d, self.char_dict, reverse_order=True, append_eos=True
        )
        tgt_dataset = char_data.InMemoryNumpyWordCharDataset()
        tgt_dataset.parse(
            self.trg_txt, self.d, self.char_dict, reverse_order=True, append_eos=True
        )
        char_dataset = char_data.LanguagePairCharDataset(
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            src_dict=self.d,
            tgt=tgt_dataset,
            tgt_sizes=tgt_dataset.sizes,
            tgt_dict=self.d,
        )
        samples = [char_dataset[i] for i in range(len(char_dataset))]
        collate_data = char_dataset.collater(samples)
        ids = collate_data["id"]
        ntokens = collate_data["ntokens"]
        assert len(ids) == 4
        assert ntokens == 32
        net_input = collate_data["net_input"]
        assert net_input["char_inds"].size() == torch.Size([4, 11, 4])
        assert net_input["prev_output_chars"].size() == torch.Size([4, 12, 4])
        assert collate_data["target_char_inds"].size() == torch.Size([4, 11, 4])
        assert net_input["prev_output_word_lengths"].size() == torch.Size([4, 12])
        for i in range(net_input["prev_output_chars"].size()[0]):
            assert net_input["prev_output_chars"][i, 0, 0] == self.d.eos_index
            # Asseting that the generated word before the first was is only eos.
            assert net_input["prev_output_word_lengths"][i][0] == 1

    def test_collate_char_dataset_w_unk(self):
        """
        We intentionally intorduce a text that has unknown words in it.
        """

        # srct and trgt are unknown words.
        src_txt = test_utils.write_lines_to_temp_file(
            ["srcA srct srcA srcA", "srcA srcA srcA srcA"]
        )
        trg_txt = test_utils.write_lines_to_temp_file(
            ["trgA trgA trgt trgB", "trgA trgB trgC trgD"]
        )
        src_dataset = char_data.InMemoryNumpyWordCharDataset()
        src_dataset.parse(
            src_txt, self.d, self.char_dict, reverse_order=True, append_eos=True
        )
        tgt_dataset = char_data.InMemoryNumpyWordCharDataset(ignore_chars_for_unks=True)
        tgt_dataset.parse(
            trg_txt, self.d, self.char_dict, reverse_order=True, append_eos=True
        )

        # Confirming that the third word in an unknown.
        assert tgt_dataset.char_offsets[1] + 1 == tgt_dataset.char_offsets[2]
        assert (
            tgt_dataset.char_buffer[tgt_dataset.char_offsets[1]] == self.char_dict.eos()
        )

        char_dataset = char_data.LanguagePairCharDataset(
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            src_dict=self.d,
            tgt=tgt_dataset,
            tgt_sizes=tgt_dataset.sizes,
            tgt_dict=self.d,
        )
        samples = [char_dataset[i] for i in range(len(char_dataset))]
        collate_data = char_dataset.collater(samples)
        ids = collate_data["id"]
        assert len(ids) == 2
        net_input = collate_data["net_input"]

        assert net_input["prev_output_word_lengths"][0][2] == 1
        assert torch.equal(
            net_input["prev_output_chars"][0][2], torch.tensor([2, 0, 0, 0])
        )
        assert torch.equal(
            collate_data["target_char_inds"][0][1], torch.tensor([2, 0, 0, 0])
        )

    def test_collate_char_dataset_no_tgt(self):
        src_dataset = char_data.InMemoryNumpyWordCharDataset()
        src_dataset.parse(
            self.src_txt, self.d, self.char_dict, reverse_order=True, append_eos=True
        )
        char_dataset = char_data.LanguagePairCharDataset(
            src=src_dataset, src_sizes=src_dataset.sizes, src_dict=self.d
        )
        samples = [char_dataset[i] for i in range(len(char_dataset))]
        collate_data = char_dataset.collater(samples)
        ids = collate_data["id"]
        ntokens = collate_data["ntokens"]
        assert len(ids) == 4
        assert ntokens is None
        net_input = collate_data["net_input"]
        assert net_input["char_inds"].size() == torch.Size([4, 11, 4])
        assert net_input["prev_output_chars"] is None
        assert collate_data["target_char_inds"] is None
        assert net_input["prev_output_word_lengths"] is None
