#!/usr/bin/env python3

import argparse
import numpy as np
import os
import torch

from fairseq import data, indexed_dataset, tokenizer
from typing import NamedTuple, Optional

from pytorch_translate import dictionary as pytorch_translate_dictionary


class CorpusConfig(NamedTuple):
    dialect: str
    data_file: str


class ParallelCorpusConfig(NamedTuple):
    source: CorpusConfig
    target: CorpusConfig


class InMemoryNumpyDataset(indexed_dataset.IndexedDataset):
    """analogous to fairseq.indexed_dataset.IndexedInMemoryDataset"""

    def __init__(self):
        """Initialize empty dataset"""
        self.buffer = None
        self.offsets = None
        self.sizes = None

    def __getitem__(self, i):
        assert i < self.__len__(), f"index {i} out of range!"
        a = self.buffer[self.offsets[i] : self.offsets[i + 1]]
        return torch.from_numpy(a)

    def __len__(self):
        # offsets includes 0 and end indices for each example
        return self.offsets.size - 1

    def __del__(self):
        pass

    def save(self, path):
        assert self.buffer is not None
        assert self.offsets is not None
        np.savez(path, buffer=self.buffer, offsets=self.offsets)

    def load(self, path):
        npz = np.load(path)
        self.buffer = npz["buffer"]
        self.offsets = npz["offsets"]
        self.sizes = self.offsets[1:] - self.offsets[:-1]

    def parse(self, path, dict, reverse_order=False, append_eos=False):
        array_list = []
        offsets = [0]
        sizes = []
        with open(path, "r") as f:
            for line in f:
                words = tokenizer.tokenize_line(line)
                if reverse_order:
                    words.reverse()
                inds = [dict.index(w) for w in words]
                if append_eos:
                    inds.append(dict.eos_index)

                array_list.append(np.array(inds, dtype=np.int32))
                offsets.append(offsets[-1] + len(inds))
                sizes.append(len(inds))

        # +1 for Lua compatibility
        self.buffer = np.concatenate(array_list) + 1
        self.offsets = np.array(offsets, dtype=np.int32)
        self.sizes = np.array(sizes, dtype=np.int32)
        del array_list
        del offsets
        del sizes

    @staticmethod
    def create_from_file(path):
        result = InMemoryNumpyDataset()
        result.load(path)
        return result


def make_language_pair_dataset_from_text(
    source_text_file: str,
    target_text_file: str,
    source_dict: pytorch_translate_dictionary.Dictionary,
    target_dict: pytorch_translate_dictionary.Dictionary,
    append_eos: Optional[bool] = False,
    reverse_source: Optional[bool] = True,
) -> data.LanguagePairDataset:
    return data.LanguagePairDataset(
        src=indexed_dataset.IndexedRawTextDataset(
            path=source_text_file,
            dictionary=source_dict,
            append_eos=append_eos,
            reverse_order=reverse_source,
        ),
        dst=indexed_dataset.IndexedRawTextDataset(
            path=target_text_file,
            dictionary=target_dict,
            # We always append EOS to the target sentence since we still want
            # the model to output an indication the sentence has finished, even
            # if we don't append the EOS symbol to the source sentence
            # (to prevent the model from misaligning UNKs or other words
            # to the frequently occurring EOS).
            append_eos=True,
            # We don't reverse the order of the target sentence, since
            # even if the source sentence is fed to the model backwards,
            # we still want the model to start outputting from the first word.
            reverse_order=False,
        ),
        pad_idx=source_dict.pad(),
        eos_idx=source_dict.eos(),
    )


def load_binarized_dataset(
    train_corpus: ParallelCorpusConfig,
    eval_corpus: ParallelCorpusConfig,
    train_split: str,
    eval_split: str,
    args: argparse.Namespace,
) -> data.LanguageDatasets:
    source_dict = pytorch_translate_dictionary.Dictionary.load(args.source_vocab_file)
    target_dict = pytorch_translate_dictionary.Dictionary.load(args.target_vocab_file)

    dataset = data.LanguageDatasets(
        src=train_corpus.source.dialect,
        dst=train_corpus.target.dialect,
        src_dict=source_dict,
        dst_dict=target_dict,
    )

    for split, corpus in [(train_split, train_corpus), (eval_split, eval_corpus)]:
        if not os.path.exists(corpus.source.data_file):
            raise ValueError(f"{corpus.source.data_file} for {split} not found!")
        if not os.path.exists(corpus.target.data_file):
            raise ValueError(f"{corpus.target.data_file} for {split} not found!")

        dataset.splits[split] = data.LanguagePairDataset(
            src=InMemoryNumpyDataset.create_from_file(corpus.source.data_file),
            dst=InMemoryNumpyDataset.create_from_file(corpus.target.data_file),
            pad_idx=source_dict.pad(),
            eos_idx=source_dict.eos(),
        )

    return dataset
