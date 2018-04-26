#!/usr/bin/env python3

import argparse
import os

from fairseq import data, indexed_dataset
from typing import NamedTuple, Optional

from fbtranslate import dictionary as fbtranslate_dictionary


class CorpusConfig(NamedTuple):
    dialect: str
    data_file: str


class ParallelCorpusConfig(NamedTuple):
    source: CorpusConfig
    target: CorpusConfig


def make_language_pair_dataset(
    source_file: str,
    target_file: str,
    source_dict: fbtranslate_dictionary.Dictionary,
    target_dict: fbtranslate_dictionary.Dictionary,
    args: Optional[argparse.Namespace] = None,
) -> data.LanguagePairDataset:
    return data.LanguagePairDataset(
        src=indexed_dataset.IndexedRawTextDataset(
            path=source_file,
            dictionary=source_dict,
            append_eos=args.append_eos_to_source if args is not None else True,
            reverse_order=args.reverse_source if args is not None else False,
        ),
        dst=indexed_dataset.IndexedRawTextDataset(
            path=target_file,
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


def load_raw_text_dataset(
    train_corpus: ParallelCorpusConfig,
    eval_corpus: ParallelCorpusConfig,
    train_split: str,
    eval_split: str,
    args: argparse.Namespace,
    penalized_target_tokens_file=None,
) -> data.LanguageDatasets:
    source_dict = fbtranslate_dictionary.Dictionary.load(args.source_vocab_file)
    target_dict = fbtranslate_dictionary.Dictionary.load(args.target_vocab_file)

    dataset = data.LanguageDatasets(
        src=train_corpus.source.dialect,
        dst=train_corpus.target.dialect,
        src_dict=source_dict,
        dst_dict=target_dict,
    )

    prev_source_dialect = None
    prev_target_dialect = None

    for split, corpus in [
        (train_split, train_corpus),
        (eval_split, eval_corpus),
    ]:
        # Sanity check that all language directions are consistent until this
        # has been updated to support multilingual corpora.
        if prev_source_dialect is None and prev_target_dialect is None:
            prev_source_dialect = corpus.source.dialect
            prev_target_dialect = corpus.target.dialect
        elif (prev_source_dialect != corpus.source.dialect or
                prev_target_dialect != corpus.target.dialect):
            raise ValueError(
                'We currently only support monolingual directions - expected '
                '{}->{} for all corpora, but found {}->{} for split {}'.format(
                    prev_source_dialect,
                    prev_target_dialect,
                    corpus.source.dialect,
                    corpus.target.dialect,
                    split,
                )
            )

        dataset.splits[split] = make_language_pair_dataset(
            source_file=corpus.source.data_file,
            target_file=corpus.target.data_file,
            source_dict=source_dict,
            target_dict=target_dict,
            args=args,
        )
    return dataset


def build_vocab_from_corpus(
    corpus_file: str,
    dialect: str,
    save_dir: str,
    max_vocab_size: int,
    tokens_with_penalty: Optional[str] = None,
):
    vocab_file = os.path.join(save_dir, f'dictionary-{dialect}.txt')
    d = fbtranslate_dictionary.Dictionary()
    with open(corpus_file, 'r') as f:
        for line in f:
            tokens = line.split()
            for t in tokens:
                token_index = d.add_symbol(t)

    # Set indices to receive penalty
    if tokens_with_penalty:
        # Assume input tokens are unique
        bad_words_list = []
        with open(tokens_with_penalty, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) == 1:
                    bad_words_list.append(tokens[0])

        for token, token_index in d.indices.items():
            if token in bad_words_list:
                d.profanity_indices.add(token_index)

    d.finalize()
    d.save(vocab_file, threshold=0, nwords=max_vocab_size)
    print(f'Generated new vocab file saved at {vocab_file}.')
    if max_vocab_size < 0:
        print('No maximum vocab sized enforced.')
    else:
        print(f'Maximum vocab size {max_vocab_size}')

    return vocab_file


def build_vocab_if_nonexistent(
    vocab_file: str,
    corpus_file: str,
    dialect: str,
    save_dir: str,
    max_vocab_size: int,
    tokens_with_penalty: str = None,
):
    if vocab_file and os.path.isfile(vocab_file):
        return vocab_file
    # Vocab file is either unspecified or does not exist
    if vocab_file and not os.path.isfile(vocab_file):
        print(f'Vocab file {vocab_file} does not exist.')
    return build_vocab_from_corpus(
        corpus_file=corpus_file,
        dialect=dialect,
        save_dir=save_dir,
        max_vocab_size=max_vocab_size,
        tokens_with_penalty=tokens_with_penalty,
    )
