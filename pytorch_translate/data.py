#!/usr/bin/env python3

import argparse
import numpy as np
import os
import torch

from fairseq import data, indexed_dataset, tokenizer
from typing import NamedTuple, Optional

from pytorch_translate import char_data
from pytorch_translate import dictionary as pytorch_translate_dictionary


# The n-th source|target language is represented with the token
# n+MULTILING_DIALECT_ID_OFFSET in the source|target token sequence.
MULTILING_DIALECT_ID_OFFSET = 10


class CorpusConfig(NamedTuple):
    dialect: str
    data_file: str


class MultilingualCorpusConfig(NamedTuple):
    dialect_id: Optional[int]
    data_file: str
    dict: pytorch_translate_dictionary.Dictionary
    oversampling: int


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
        self.parse_multilingual(
            [
                MultilingualCorpusConfig(
                    dialect_id=None, data_file=path, dict=dict, oversampling=1
                )
            ],
            reverse_order=reverse_order,
            append_eos=append_eos,
        )

    def parse_multilingual(
        self, corpora, reverse_order=False, append_eos=False, prepend_language_id=True
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
                    words = tokenizer.tokenize_line(line)
                    if reverse_order:
                        words.reverse()
                    inds = (
                        prepend_inds
                        + [corpus_config.dict.index(w) for w in words]
                        + append_inds
                    )
                    for _ in range(corpus_config.oversampling):
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


def is_multilingual(args):
    if hasattr(args, "multiling_encoder_lang"):
        return bool(args.multiling_encoder_lang)
    return args.multiling_source_lang_id is not None


def make_language_pair_dataset_from_text(
    source_text_file: str,
    target_text_file: str,
    source_dict: pytorch_translate_dictionary.Dictionary,
    target_dict: pytorch_translate_dictionary.Dictionary,
    append_eos: Optional[bool] = False,
    reverse_source: Optional[bool] = True,
    char_source_dict: Optional[pytorch_translate_dictionary.Dictionary] = None,
) -> data.LanguagePairDataset:
    dst_dataset = indexed_dataset.IndexedRawTextDataset(
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
    )

    if char_source_dict is not None:
        src_dataset = char_data.InMemoryNumpyWordCharDataset()
        src_dataset.parse(
            path=source_text_file,
            word_dict=source_dict,
            char_dict=char_source_dict,
            reverse_order=reverse_source,
            append_eos=append_eos,
        )
        return char_data.LanguagePairSourceCharDataset(
            src=src_dataset,
            dst=dst_dataset,
            pad_idx=source_dict.pad(),
            eos_idx=source_dict.eos(),
        )
    else:
        return data.LanguagePairDataset(
            src=indexed_dataset.IndexedRawTextDataset(
                path=source_text_file,
                dictionary=source_dict,
                append_eos=append_eos,
                reverse_order=reverse_source,
            ),
            dst=dst_dataset,
            pad_idx=source_dict.pad(),
            eos_idx=source_dict.eos(),
        )


class IndexedRawTextDatasetWithLangId(indexed_dataset.IndexedRawTextDataset):
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
        lang_id_tensor = torch.IntTensor(
            [self.lang_id + MULTILING_DIALECT_ID_OFFSET + 1]
        )  # +1 for Lua compatibility

        def add_lang_id(tokens):
            if self.prepend_language_id:
                return torch.cat([lang_id_tensor, tokens])
            return torch.cat([tokens, lang_id_tensor])

        self.tokens_list = [add_lang_id(t) for t in self.tokens_list]


def make_language_pair_dataset_from_text_multilingual(
    source_text_file: str,
    target_text_file: str,
    source_lang_id: int,
    target_lang_id: int,
    source_dict: pytorch_translate_dictionary.Dictionary,
    target_dict: pytorch_translate_dictionary.Dictionary,
    append_eos: Optional[bool] = False,
    reverse_source: Optional[bool] = True,
) -> data.LanguagePairDataset:
    return data.LanguagePairDataset(
        src=IndexedRawTextDatasetWithLangId(
            path=source_text_file,
            dictionary=source_dict,
            lang_id=source_lang_id,
            append_eos=append_eos,
            reverse_order=reverse_source,
            prepend_language_id=False,
        ),
        dst=IndexedRawTextDatasetWithLangId(
            path=target_text_file,
            dictionary=target_dict,
            lang_id=target_lang_id,
            append_eos=True,
            reverse_order=False,
            prepend_language_id=True,
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
    use_char_source: bool = False,
) -> data.LanguageDatasets:
    if is_multilingual(args):  # Dummy dictionaries
        source_dict = pytorch_translate_dictionary.Dictionary()
        target_dict = pytorch_translate_dictionary.Dictionary()
    else:
        source_dict = pytorch_translate_dictionary.Dictionary.load(
            args.source_vocab_file
        )
        target_dict = pytorch_translate_dictionary.Dictionary.load(
            args.target_vocab_file
        )

    if use_char_source:
        char_source_dict = pytorch_translate_dictionary.Dictionary.load(
            args.char_source_vocab_file
        )
        # this attribute is used for CharSourceModel construction
        args.char_source_dict_size = len(char_source_dict)

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

        dst_dataset = InMemoryNumpyDataset.create_from_file(corpus.target.data_file)
        if use_char_source:
            src_dataset = char_data.InMemoryNumpyWordCharDataset.create_from_file(
                corpus.source.data_file
            )
            dataset.splits[split] = char_data.LanguagePairSourceCharDataset(
                src=src_dataset,
                dst=dst_dataset,
                pad_idx=source_dict.pad(),
                eos_idx=source_dict.eos(),
            )
        else:
            src_dataset = InMemoryNumpyDataset.create_from_file(corpus.source.data_file)
            dataset.splits[split] = data.LanguagePairDataset(
                src=src_dataset,
                dst=dst_dataset,
                pad_idx=source_dict.pad(),
                eos_idx=source_dict.eos(),
            )

    return dataset
