#!/usr/bin/env python3

import os
import tempfile
from typing import Dict, NamedTuple, Optional

import numpy as np
import torch
from fairseq import data, tokenizer
from pytorch_translate import dictionary as pytorch_translate_dictionary


# The n-th source|target language is represented with the token
# n+MULTILING_DIALECT_ID_OFFSET in the source|target token sequence.
MULTILING_DIALECT_ID_OFFSET = 10

# Read bigger arrays from disc instead of memory
ARRAY_SIZE_LIMIT_FOR_MEMORY = 10 ** 10  # 10GB


class CorpusConfig(NamedTuple):
    dialect: str
    data_file: str


class MultilingualCorpusConfig(NamedTuple):
    dialect_id: Optional[int]
    data_file: str
    dict: pytorch_translate_dictionary.Dictionary
    oversampling: int
    char_dict: Optional[pytorch_translate_dictionary.Dictionary] = None


class ParallelCorpusConfig(NamedTuple):
    source: CorpusConfig
    target: CorpusConfig
    weights_file: Optional[str]


class ParallelCorporaMapConfig(NamedTuple):
    src_files: Dict[str, str]
    tgt_files: Dict[str, str]


class InMemoryNumpyDataset(data.indexed_dataset.IndexedDataset):
    """analogous to fairseq.data.indexed_dataset.IndexedCachedDataset"""

    def __init__(self):
        """Initialize empty dataset"""
        self.buffer = None
        self.offsets = None
        self.sizes = None

    def __getitem__(self, i):
        assert i < self.__len__(), f"index {i} out of range!"
        a = self.buffer[self.offsets[i] : self.offsets[i + 1]]
        return torch.from_numpy(a).long()

    def __len__(self):
        # offsets includes 0 and end indices for each example
        return self.offsets.size - 1

    def __del__(self):
        if isinstance(self.buffer, np.memmap):
            os.remove(self.buffer.filename)

    def save(self, path):
        assert self.buffer is not None
        assert self.offsets is not None
        np.savez(path, buffer=self.buffer, offsets=self.offsets)

    def load(self, path, num_examples_limit: Optional[int] = None):
        npz = np.load(path)

        # For big input data, we don't want the cpu to OOM.
        # Therefore, we are loading the huge buffer array into disc
        # and reading it from disc instead of memory.
        if npz["buffer"].nbytes > ARRAY_SIZE_LIMIT_FOR_MEMORY:
            self.buffer = np.memmap(
                tempfile.NamedTemporaryFile().name,
                dtype="float32",
                mode="w+",
                shape=npz["buffer"].shape,
            )
            self.buffer[:] = npz["buffer"][:]
        else:
            self.buffer = npz["buffer"]
        self.offsets = npz["offsets"]
        if num_examples_limit is not None and len(self.offsets) > num_examples_limit:
            self.offsets = self.offsets[: num_examples_limit + 1]
            self.buffer = self.buffer[: self.offsets[-1]]
        self.sizes = self.offsets[1:] - self.offsets[:-1]

    def parse(
        self,
        path,
        dictionary,
        reverse_order=False,
        append_eos=False,
        already_numberized=False,
    ):
        self.parse_multilingual(
            [
                MultilingualCorpusConfig(
                    dialect_id=None, data_file=path, dict=dictionary, oversampling=1
                )
            ],
            reverse_order=reverse_order,
            append_eos=append_eos,
            already_numberized=already_numberized,
        )

    def parse_multilingual(
        self,
        corpora,
        reverse_order=False,
        append_eos=False,
        prepend_language_id=True,
        already_numberized=False,
    ):
        """Add sentences from text files to the dataset.

        This method reads pairs of text files containing source and target
        sides of a bitext. Sentences are converted to integer sequences by
        tokenization and dictionary look-up. Note that this method removes all
        sentences which have been previously added to the data set.

        Example (single sentence):
            token_sequence = [123, 234, 345]
            dict.eos_idx = 2
            dialect_id = 10
            Result:
                reverse_order=False, append_eos=True, prepend_language_id=True:
                    [10, 123, 234, 345, 2]
                reverse_order=False, append_eos=True, prepend_language_id=False:
                    [123, 234, 345, 2, 10]
                reverse_order=True, append_eos=True, prepend_language_id=True:
                    [10, 345, 234, 123, 2]
                reverse_order=True, append_eos=True, prepend_language_id=False:
                    [345, 234, 123, 2, 10]

        Args:
            corpora: List of MultilingualCorpusConfig. If dialect_id is not
                None, it is added to the token sequence.
            reverse_order (bool): Whether to reverse the integer token sequence.
            append_eos (bool): Whether to add the end-of-sentence symbol to each
                sentence.
            prepend_language_id (bool): Only used if dialect_id is not None. If
                true, add ID at the begin of the token sequence. Otherwise, add
                it at the end of the token sequence.
            already_numberized (bool): If data_file contains lines of
                numberized tokens, then already_numberized should be set to True
                If data_file contains raw text sentences, then
                already_numberized should be False (default) -- in which case
                each line is tokenized with tokenizer then numberized with the
                dictionary before being added to the output buffer.

        """
        array_list = []
        offsets = [0]
        sizes = []
        for corpus_config in corpora:
            prepend_inds = []
            append_inds = []
            if append_eos:
                append_inds.append(corpus_config.dict.eos_index)
            if corpus_config.dialect_id is not None:
                if prepend_language_id:
                    prepend_inds.append(corpus_config.dialect_id)
                else:
                    append_inds.append(corpus_config.dialect_id)
            with open(corpus_config.data_file, "r") as f:
                for line in f:
                    if already_numberized:
                        inds = line.strip().split()
                        inds = [int(ind) for ind in inds]
                    else:
                        words = tokenizer.tokenize_line(line)
                        inds = [corpus_config.dict.index(w) for w in words]

                    if reverse_order:
                        inds.reverse()
                    inds = prepend_inds + inds + append_inds
                    for _ in range(corpus_config.oversampling):
                        array_list.append(np.array(inds, dtype=np.int32))
                        offsets.append(offsets[-1] + len(inds))
                        sizes.append(len(inds))

        self.buffer = np.concatenate(array_list)
        self.offsets = np.array(offsets, dtype=np.int64)
        self.sizes = np.array(sizes, dtype=np.int32)
        del array_list
        del offsets
        del sizes

    def load_from_sequences(self, sequences):
        """
        Load data set from a list of sequences, each a list or numpy array of
        indices. Note that this method removes all sentences which have been
        previously added to the data set.
        """
        array_list = []
        offsets = [0]
        sizes = []
        for inds in sequences:
            array_list.append(np.array(inds, dtype=np.int32))
            offsets.append(offsets[-1] + len(inds))
            sizes.append(len(inds))

        self.buffer = np.concatenate(array_list)
        self.offsets = np.array(offsets, dtype=np.int32)
        self.sizes = np.array(sizes, dtype=np.int32)
        del array_list
        del offsets
        del sizes

    @staticmethod
    def create_from_file(path, num_examples_limit: Optional[int] = None):
        result = InMemoryNumpyDataset()
        result.load(path, num_examples_limit=num_examples_limit)
        return result


def is_multilingual(args):
    if hasattr(args, "multiling_encoder_lang"):
        return bool(args.multiling_encoder_lang)
    return args.multiling_source_lang is not None


class IndexedRawTextDatasetWithLangId(data.IndexedRawTextDataset):
    """Adds language IDs to an IndexedRawTextDataset"""

    def __init__(
        self,
        path,
        dictionary,
        lang_id,
        append_eos=True,
        reverse_order=False,
        prepend_language_id=True,
    ):
        self.lang_id = lang_id
        self.prepend_language_id = prepend_language_id
        super(IndexedRawTextDatasetWithLangId, self).__init__(
            path=path,
            dictionary=dictionary,
            append_eos=append_eos,
            reverse_order=reverse_order,
        )

    def read_data(self, path, dictionary):
        super(IndexedRawTextDatasetWithLangId, self).read_data(path, dictionary)
        # Postprocess self.tokens_list and self.sizes
        self.sizes += 1
        lang_id_tensor = torch.LongTensor([self.lang_id + MULTILING_DIALECT_ID_OFFSET])

        def add_lang_id(tokens):
            if self.prepend_language_id:
                return torch.cat([lang_id_tensor, tokens])
            return torch.cat([tokens, lang_id_tensor])

        self.tokens_list = [add_lang_id(t) for t in self.tokens_list]
