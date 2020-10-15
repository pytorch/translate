#!/usr/bin/env python3

import os
import shutil
import tempfile
from typing import Dict, NamedTuple, Optional

import numpy as np
import torch
from fairseq import data, tokenizer
from fvcore.common.file_io import PathManager
from pytorch_translate import constants
from pytorch_translate.data import dictionary as pytorch_translate_dictionary


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
    weights_file: Optional[str] = None


class ParallelCorporaMapConfig(NamedTuple):
    src_files: Dict[str, str]
    tgt_files: Dict[str, str]


class InMemoryIndexedDataset(data.indexed_dataset.IndexedDataset):
    """analogous to fairseq.data.indexed_dataset.IndexedCachedDataset.
    Support loading .idx + .bin as fairseq does and also .npz. In self
    initializer if
    (1) path is passed in: call the initializer of parent class and loads
    idx + bin from the path.
    (2) path is None: Initialize an empty class. Call load(path) to load
    .npz data.
    When self.is_npz=True, use the implementation in this class which overrides
    parent methods. Otherwise use parent functions in fairseq."""

    def __init__(self, path=None, fix_lua_indexing=False, read_data=True):
        if path is None:
            self.buffer = None
            self.offsets = None
            self.sizes = None
            self.is_npz = True
        else:
            self.is_npz = False
            super().__init__(path, fix_lua_indexing)

    def __getitem__(self, i):
        if self.is_npz:
            assert i < self.__len__(), f"index {i} out of range!"
            a = self.buffer[self.offsets[i] : self.offsets[i + 1]]
            return torch.from_numpy(a).long()
        else:
            return super().__getitem__(i)

    def __len__(self):
        if self.is_npz:
            # offsets includes 0 and end indices for each example
            return self.offsets.size - 1
        else:
            return super().__len__()

    def __del__(self):
        if self.is_npz:
            if isinstance(self.buffer, np.memmap):
                os.remove(self.buffer.filename)
        else:
            super().__del__()

    def save(self, path):
        assert self.buffer is not None
        assert self.offsets is not None
        with PathManager.open(path, "wb") as f:
            np.savez(f, buffer=self.buffer, offsets=self.offsets)

    def reverse(self, eos_token=True):
        for i in range(len(self.offsets) - 1):
            start_offset = self.offsets[i]
            end_offset = self.offsets[i + 1] - 1 if eos_token else self.offsets[i + 1]

            self.buffer[start_offset:end_offset] = self.buffer[start_offset:end_offset][
                ::-1
            ]

    def load(self, path, num_examples_limit: Optional[int] = None):
        with PathManager.open(path, "rb") as f:
            npz = np.load(f)

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
        print(corpora)
        for corpus_config in corpora:
            print(corpus_config)
            print(corpus_config.data_file)
            prepend_inds = []
            append_inds = []
            if append_eos:
                append_inds.append(corpus_config.dict.eos_index)
            if corpus_config.dialect_id is not None:
                if prepend_language_id:
                    prepend_inds.append(corpus_config.dialect_id)
                else:
                    append_inds.append(corpus_config.dialect_id)
            with PathManager.open(corpus_config.data_file, "r") as f:
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
    def create_from_file(path, is_npz=True, num_examples_limit: Optional[int] = None):
        if is_npz:
            # npz format
            result = InMemoryIndexedDataset()
            result.load(path, num_examples_limit=num_examples_limit)
            return result
        else:
            # idx, bin format
            if path.startswith("manifold://"):
                tempdir = tempfile.mkdtemp()
                basename = os.path.basename(path)
                prefix = os.path.join(tempdir, basename)
                for suffix in ("bin", "idx"):
                    remote_path = f"{path}.{suffix}"
                    local_path = PathManager.get_local_path(remote_path)
                    shutil.copy(local_path, f"{prefix}.{suffix}")
                path = prefix
            return data.indexed_dataset.MMapIndexedDataset(path)

    def subsample(self, indices):
        """
        Subsample dataset to include only those items indexed by input
        argument indices.
        """
        array_list = []
        offsets = [0]
        sizes = []
        for i in indices:
            array_list.append(self[i])
            offsets.append(offsets[-1] + len(array_list[-1]))
            sizes.append(len(array_list[-1]))
        self.buffer = np.concatenate(array_list)
        self.offsets = np.array(offsets, dtype=np.int32)
        self.sizes = np.array(sizes, dtype=np.int32)


def is_multilingual_many_to_one(args):
    """
    Checks whether we would be using the multilingual implementation in
    pytorch_translate/multilingual.py. This is currently used as the default
    many-to-one multilingual architecture.
    """
    if hasattr(args, "multiling_encoder_lang"):
        return bool(args.multiling_encoder_lang)
    return args.multiling_source_lang is not None


def is_multilingual(args):
    """
    Checks whether we would be using the multilingual implementation that
    extends fariseq's MultilingualTranslationTask and FairseqMultiModel
    """
    return args.task == constants.MULTILINGUAL_TRANSLATION_TASK


def is_latent_variable(args):
    return args.task == constants.LATENT_VARIABLE_TASK


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


def subsample_pair_dataset(dataset, num_samples, seed_num=-1):
    if len(dataset) <= num_samples:
        return
    if seed_num != -1:
        np.random.seed(seed_num)
    indices = np.random.permutation(len(dataset))[:num_samples]
    dataset.src.subsample(indices)
    dataset.src_sizes = dataset.src.sizes
    dataset.tgt.subsample(indices)
    dataset.tgt_sizes = dataset.tgt.sizes
