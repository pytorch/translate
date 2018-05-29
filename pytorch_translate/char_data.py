#!/usr/bin/env python3

import numpy as np
import torch

from fairseq import data, indexed_dataset, tokenizer

from pytorch_translate.dictionary import TAGS


class InMemoryNumpyWordCharDataset(indexed_dataset.IndexedDataset):
    """analogous to fairseq.indexed_dataset.IndexedInMemoryDataset"""

    def __init__(self):
        """Initialize empty dataset"""
        self.word_buffer = None
        self.word_offsets = None
        self.char_buffer = None
        self.char_offsets = None
        self.sizes = None

    def get_tokens(self, i):
        """Get tensor of token indices for example i"""
        assert i < self.__len__(), f"index {i} out of range!"
        a = self.word_buffer[self.word_offsets[i] : self.word_offsets[i + 1]]
        return torch.from_numpy(a)

    def get_chars_list(self, i):
        """Get list of tensors of character indices for example i"""
        result = []
        for word_index in range(self.word_offsets[i], self.word_offsets[i + 1]):
            char_indices = self.char_buffer[
                self.char_offsets[word_index] : self.char_offsets[word_index + 1]
            ]
            result.append(torch.from_numpy(char_indices))
        return result

    def __len__(self):
        # offsets includes 0 and end indices for each example
        return self.word_offsets.size - 1

    def __del__(self):
        pass

    def save(self, path):
        assert self.word_buffer is not None
        assert self.word_offsets is not None
        assert self.char_buffer is not None
        assert self.char_offsets is not None
        np.savez(
            path,
            word_buffer=self.word_buffer,
            word_offsets=self.word_offsets,
            char_buffer=self.char_buffer,
            char_offsets=self.char_offsets,
        )

    def load(self, path):
        npz = np.load(path)
        if "char_buffer" not in npz or "char_offsets" not in npz:
            raise RuntimeError(f"{path} does not appear to be a word-char dataset!")
        self.word_buffer = npz["word_buffer"]
        self.word_offsets = npz["word_offsets"]
        self.sizes = self.word_offsets[1:] - self.word_offsets[:-1]
        self.char_buffer = npz["char_buffer"]
        self.char_offsets = npz["char_offsets"]

    def parse(self, path, word_dict, char_dict, reverse_order=False, append_eos=False):
        word_array_list = []
        word_offsets = [0]
        char_array_list = []
        char_offsets = [0]
        sizes = []
        with open(path, "r") as f:
            for line in f:
                words = tokenizer.tokenize_line(line)
                if reverse_order:
                    words.reverse()
                word_inds = [word_dict.index(w) for w in words]
                if append_eos:
                    word_inds.append(dict.eos_index)

                word_array_list.append(np.array(word_inds, dtype=np.int32))
                word_offsets.append(word_offsets[-1] + len(word_inds))
                sizes.append(len(word_inds))

                for word in words:
                    chars = [word] if word in TAGS else list(word)
                    char_inds = [char_dict.index(c) for c in chars]
                    char_array_list.append(np.array(char_inds, dtype=np.int32))
                    char_offsets.append(char_offsets[-1] + len(char_inds))

        # +1 for Lua compatibility
        self.word_buffer = np.concatenate(word_array_list) + 1
        self.word_offsets = np.array(word_offsets, dtype=np.int32)
        self.char_buffer = np.concatenate(char_array_list)
        self.char_offsets = np.array(char_offsets, dtype=np.int32)
        self.sizes = np.array(sizes, dtype=np.int32)

        del word_array_list, word_offsets, char_array_list, char_offsets, sizes

    @staticmethod
    def create_from_file(path):
        result = InMemoryNumpyWordCharDataset()
        result.load(path)
        return result


class LanguagePairSourceCharDataset(torch.utils.data.Dataset):
    """
    Version of fairseq.data.LanguagePairDataset which represents source
    sentences as sequences of words, each represented as a sequence of
    characters (with numberized indices for both words and characters).
    Right-padded only.
    """

    def __init__(self, src, dst, pad_idx, eos_idx):
        """
        src : InMemoryNumpyWordCharDataset
        dst : InMemoryNumpyDataset
        """
        self.src = src
        self.dst = dst
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

    def __getitem__(self, i):
        example = {
            "id": i,
            "source_tokens": self.src.get_tokens(i).long(),
            "source_chars_list": self.src.get_chars_list(i),
        }
        if self.dst:
            # subtract 1 for 0-based indexing (fairseq legacy)
            example["target"] = self.dst[i].long() - 1

        return example

    def __len__(self):
        """Length in words"""
        return len(self.src)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        # sort in order of descending number of words
        samples.sort(key=lambda s: len(s["source_tokens"]), reverse=True)
        max_words = len(samples[0]["source_tokens"])

        id = torch.LongTensor([s["id"] for s in samples])
        src_lengths = torch.LongTensor([len(s["source_tokens"]) for s in samples])

        word_lengths = torch.LongTensor(len(samples), max_words).fill_(0)
        for i, s in enumerate(samples):
            word_lengths_array = np.array([len(w) for w in s["source_chars_list"]])
            word_lengths[i, : word_lengths_array.size] = torch.LongTensor(
                word_lengths_array
            )
        max_word_length = int(word_lengths.max())

        src_tokens = (
            samples[0]["source_tokens"].new(len(samples), max_words).fill_(self.pad_idx)
        )

        char_inds = (
            samples[0]["source_chars_list"][0].new(
                len(samples), max_words, max_word_length
            )
            .long()
            .fill_(self.pad_idx)
        )

        for i, s in enumerate(samples):
            chars_list = s["source_chars_list"]
            for j, chars in enumerate(chars_list):
                char_inds[i, j, : word_lengths[i, j]] = chars

        target = None
        prev_output_tokens = None
        ntokens = None
        if self.dst:

            def merge(key, left_pad=False, move_eos_to_beginning=False):
                return data.LanguagePairDataset.collate_tokens(
                    [s[key] for s in samples],
                    self.pad_idx,
                    self.eos_idx,
                    left_pad,
                    move_eos_to_beginning,
                )

            target = merge("target", move_eos_to_beginning=False)
            prev_output_tokens = merge("target", move_eos_to_beginning=True)

            ntokens = sum(len(s["target"]) for s in samples)

        return {
            "id": id,
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "char_inds": char_inds,
                "word_lengths": word_lengths,
                "prev_output_tokens": prev_output_tokens,
            },
            "target": target,
        }
