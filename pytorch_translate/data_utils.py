#!/usr/bin/env python3

import os

from pytorch_translate import char_data, data as ptt_data


def load_monolingual_dataset(task, bin_path, is_source=False):
    if task.args.log_verbose:
        print("Starting to load binarized monolingual data file.", flush=True)

    if not os.path.exists(bin_path):
        raise ValueError(f"Monolingual binary path {bin_path} not found!")

    if task.char_source_dict is not None and is_source:
        dataset = char_data.InMemoryNumpyWordCharDataset.create_from_file(path=bin_path)

    else:
        dataset = ptt_data.InMemoryNumpyDataset.create_from_file(path=bin_path)

    if task.args.log_verbose:
        print("Finished loading dataset", flush=True)

    print(
        f"""| Loaded {len(dataset)} monolingual examples for """
        f"""{"source" if is_source else "target"}"""
    )
    return dataset


def validate_corpus_exists(corpus: ptt_data.ParallelCorpusConfig, split: str):
    """
    Makes sure that the files in the `corpus` are valid files. `split` is used
    for logging.
    """
    if not os.path.exists(corpus.source.data_file):
        raise ValueError(f"{corpus.source.data_file} for {split} not found!")
    if not os.path.exists(corpus.target.data_file):
        raise ValueError(f"{corpus.target.data_file} for {split} not found!")
