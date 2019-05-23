#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from pytorch_translate.data import dictionary as pytorch_translate_dictionary


def get_source_langs(lang_pairs: List[str]) -> List[str]:
    """
    Return list of source languages from args.lang_pairs
    lang_pairs: List[str] where each element is a str with comma separated list
    of language pairs
    """
    return [lang_pair.split("-")[0] for lang_pair in lang_pairs]


def get_target_langs(lang_pairs: List[str]) -> List[str]:
    """
    Return list of target languages from args.lang_pairs
    lang_pairs: List[str] where each element is a str with comma separated list
    of language pairs
    """
    return [lang_pair.split("-")[1] for lang_pair in lang_pairs]


def default_binary_path(save_dir: str, lang_pair: str, lang: str, split: str) -> str:
    return os.path.join(save_dir, f"{split}-binary-{lang_pair}.{lang}")


def get_dict_paths(
    vocabulary_args: Optional[List[str]], langs: List[str], save_dir: str
) -> Dict[str, str]:
    """
    Extract dictionary files based on --vocabulary argument, for the given
    languages `langs`.
    vocabulary_arg: Optional[List[str]] where each element is a str with the format
    "lang:vocab_file"
    """
    dicts = {}
    if vocabulary_args is not None:
        for vocab_config in vocabulary_args:
            # vocab_config is in the format "lang:vocab_file"
            lang, vocab = vocab_config.split(":")
            if lang in langs:
                dicts[lang] = vocab
    for lang in langs:
        if lang not in dicts:
            dicts[lang] = pytorch_translate_dictionary.default_dictionary_path(
                save_dir=save_dir, dialect=lang
            )
    return dicts


def get_corpora_for_lang(parallel_corpora: List[str], lang: str) -> List[str]:
    """
    Fetches list of corpora that belong to given lang
    parallel_corpora: List[str] where each element is a str with the format
    "src_lang-tgt_lang:src_corpus,tgt_corpus"

    Returns [] if corpora for lang is not found
    """
    corpora = []
    for parallel_corpus_config in parallel_corpora:
        lang_pair, parallel_corpus = parallel_corpus_config.split(":")
        src_lang, tgt_lang = lang_pair.split("-")
        if src_lang == lang:
            corpora.append(parallel_corpus.split(",")[0])
        if tgt_lang == lang:
            corpora.append(parallel_corpus.split(",")[1])
    return corpora


def get_parallel_corpus_for_lang_pair(
    parallel_corpora: List[str], lang_pair: str
) -> Tuple[str, str]:
    """
    Fetches parallel corpus that belong to given lang_pair
    parallel_corpora: List[str] where each element is a str with the format
    "src_lang-tgt_lang:src_corpus,tgt_corpus"

    Returns None if parallel corpora for lang_pair is not found
    """
    for parallel_corpus_config in parallel_corpora:
        corpus_lang_pair, parallel_corpus = parallel_corpus_config.split(":")
        if corpus_lang_pair == lang_pair:
            return tuple(parallel_corpus.split(","))
    return None


def prepare_dicts(
    args: argparse.Namespace, langs: List[str]
) -> Tuple[Dict[str, str], Dict[str, pytorch_translate_dictionary.Dictionary]]:
    """
    Uses multilingual train corpora specified in args.multilingual_train_text_file
    to build dictionaries for languages specified in `langs`.
    Vocab size is defined by args.target_max_vocab_size if lang is in the set
    of target languages, otherwise it is decided by args.target_max_vocab_size
    """
    tgt_langs = get_target_langs(args.lang_pairs.split(","))
    dict_paths = get_dict_paths(args.vocabulary, langs, args.save_dir)
    lang2corpus = defaultdict(list)
    for lang in langs:
        lang2corpus[lang] = get_corpora_for_lang(
            args.multilingual_train_text_file, lang
        )
    dict_objects = {
        lang: pytorch_translate_dictionary.Dictionary.build_vocab_file_if_nonexistent(
            corpus_files=lang2corpus[lang],
            vocab_file=dict_paths[lang],
            max_vocab_size=(
                args.target_max_vocab_size
                if lang in tgt_langs
                else args.source_max_vocab_size
            ),
            tokens_with_penalty=None,
        )
        for lang in langs
    }
    return dict_paths, dict_objects
