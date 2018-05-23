#!/usr/bin/env python3

import argparse
import os

from fairseq import data, indexed_dataset
from typing import NamedTuple, Optional

from pytorch_translate import dictionary as pytorch_translate_dictionary


class CorpusConfig(NamedTuple):
    dialect: str
    data_file: str


class ParallelCorpusConfig(NamedTuple):
    source: CorpusConfig
    target: CorpusConfig


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
        if (
            not indexed_dataset.IndexedInMemoryDataset.exists(corpus.source.data_file)
            or not indexed_dataset.IndexedInMemoryDataset.exists(
                corpus.target.data_file
            )
        ):
            raise ValueError(
                f"One or both of source file: {corpus.source.data_file} and "
                f"target file: {corpus.target.data_file} for split {split} "
                f"was not found."
            )

        dataset.splits[split] = data.LanguagePairDataset(
            src=indexed_dataset.IndexedInMemoryDataset(corpus.source.data_file),
            dst=indexed_dataset.IndexedInMemoryDataset(corpus.target.data_file),
            pad_idx=source_dict.pad(),
            eos_idx=source_dict.eos(),
        )

    return dataset
