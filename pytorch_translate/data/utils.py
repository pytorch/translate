#!/usr/bin/env python3

from typing import Optional

from fvcore.common.file_io import PathManager
from pytorch_translate.data import (
    char_data,
    data as pytorch_translate_data,
    weighted_data,
)


def load_parallel_dataset(
    source_lang,
    target_lang,
    src_bin_path,
    tgt_bin_path,
    source_dictionary,
    target_dictionary,
    split,
    remove_eos_from_source,
    append_eos_to_target=True,
    char_source_dict=None,
    log_verbose=True,
):
    corpus = pytorch_translate_data.ParallelCorpusConfig(
        source=pytorch_translate_data.CorpusConfig(
            dialect=source_lang, data_file=src_bin_path
        ),
        target=pytorch_translate_data.CorpusConfig(
            dialect=target_lang, data_file=tgt_bin_path
        ),
        weights_file=None,
    )

    if log_verbose:
        print("Starting to load binarized data files.", flush=True)
    validate_corpus_exists(corpus=corpus, split=split)

    tgt_dataset = pytorch_translate_data.InMemoryIndexedDataset.create_from_file(
        corpus.target.data_file
    )
    if char_source_dict is not None:
        src_dataset = char_data.InMemoryNumpyWordCharDataset.create_from_file(
            corpus.source.data_file
        )
    else:
        src_dataset = pytorch_translate_data.InMemoryIndexedDataset.create_from_file(
            corpus.source.data_file
        )
    parallel_dataset = weighted_data.WeightedLanguagePairDataset(
        src=src_dataset,
        src_sizes=src_dataset.sizes,
        src_dict=source_dictionary,
        tgt=tgt_dataset,
        tgt_sizes=tgt_dataset.sizes,
        tgt_dict=target_dictionary,
        remove_eos_from_source=remove_eos_from_source,
        append_eos_to_target=append_eos_to_target,
    )
    return parallel_dataset, src_dataset, tgt_dataset


def load_monolingual_dataset(
    bin_path,
    is_source=False,
    char_source_dict=None,
    log_verbose=True,
    num_examples_limit: Optional[int] = None,
):
    if log_verbose:
        print("Starting to load binarized monolingual data file.", flush=True)

    if not PathManager.exists(bin_path):
        raise ValueError(f"Monolingual binary path {bin_path} not found!")

    if char_source_dict is not None and is_source:
        dataset = char_data.InMemoryNumpyWordCharDataset.create_from_file(path=bin_path)

    else:
        dataset = pytorch_translate_data.InMemoryIndexedDataset.create_from_file(
            path=bin_path, num_examples_limit=num_examples_limit
        )

    if log_verbose:
        print(f"Finished loading dataset {bin_path}", flush=True)

    print(
        f"""| Loaded {len(dataset)} monolingual examples for """
        f"""{"source" if is_source else "target"}"""
    )
    return dataset


def validate_fairseq_dataset_exists(prefix):
    if not PathManager.exists(f"{prefix}.idx"):
        raise ValueError(f"{prefix}.idx not found!")
    if not PathManager.exists(f"{prefix}.bin"):
        raise ValueError(f"{prefix}.bin not found!")


def validate_corpus_exists(
    corpus: pytorch_translate_data.ParallelCorpusConfig, split: str, is_npz: bool = True
):
    """
    Makes sure that the files in the `corpus` are valid files. `split` is used
    for logging.
    """
    if is_npz:
        if not PathManager.exists(corpus.source.data_file):
            raise ValueError(f"{corpus.source.data_file} for {split} not found!")
        if not PathManager.exists(corpus.target.data_file):
            raise ValueError(f"{corpus.target.data_file} for {split} not found!")
    else:
        validate_fairseq_dataset_exists(corpus.source.data_file)
        validate_fairseq_dataset_exists(corpus.target.data_file)
