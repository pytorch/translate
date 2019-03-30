#!/usr/bin/env python3

import numpy as np
import torch
from fairseq import data


class MultisourceLanguagePairDataset(data.LanguagePairDataset):
    """A language pair dataset with multiple source sentences for
    each target sentence."""

    def __getitem__(self, i):
        source = [src_sent.long() for src_sent in self.src[i]]
        res = {"id": i, "source": source}
        if self.tgt:
            res["target"] = self.tgt[i].long()

        return res

    def collater(self, samples):
        return MultisourceLanguagePairDataset.collate(
            samples,
            self.src_dict.pad(),
            self.src_dict.eos(),
            self.tgt is not None,
            self.left_pad_source,
            self.left_pad_target,
        )

    @staticmethod
    def collate(
        samples,
        pad_idx,
        eos_idx,
        has_target=True,
        left_pad_source=True,
        left_pad_target=False,
    ):
        if len(samples) == 0:
            return {}

        n_sources = len(samples[0]["source"])
        assert all(
            len(sample["source"]) == n_sources for sample in samples
        ), "All samples in a batch must have the same number of source sentences."

        def merge(key, left_pad, source=False, move_eos_to_beginning=False):
            if source:
                # Collate source sentences all source sentences together. Each
                return data.data_utils.collate_tokens(
                    [s[key][src_id] for s in samples for src_id in range(n_sources)],
                    pad_idx,
                    eos_idx,
                    left_pad,
                    move_eos_to_beginning,
                )
            else:
                return data.data_utils.collate_tokens(
                    [s[key] for s in samples],
                    pad_idx,
                    eos_idx,
                    left_pad,
                    move_eos_to_beginning,
                )

        id = torch.LongTensor([s["id"] for s in samples])
        src_tokens = merge("source", left_pad=left_pad_source, source=True)
        # We sort all source sentences from each batch element by length
        src_lengths = torch.LongTensor(
            [
                s["source"][src_id].numel()
                for s in samples
                for src_id in range(n_sources)
            ]
        )
        src_lengths, sort_order = src_lengths.sort(descending=True)
        # id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)
        # Record which sentence corresponds to which source and sample
        _, rev_order = sort_order.sort()
        # srcs_ids[k] contains the indices of kth source sentences of each
        # sample in src_tokens
        srcs_ids = [rev_order[k::n_sources] for k in range(n_sources)]

        prev_output_tokens = None
        target = None
        ntokens = None
        if has_target:
            target = merge("target", left_pad=left_pad_target)
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target", left_pad=left_pad_target, move_eos_to_beginning=True
            )
            ntokens = sum(len(s["target"]) for s in samples)

        return {
            "id": id,
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "src_ids": srcs_ids,
                "prev_output_tokens": prev_output_tokens,
            },
            "target": target,
        }


class IndexedRawTextMultisentDataset(data.IndexedRawTextDataset):
    """Takes a list of text file as input and binarizes them in memory at
    instantiation. Original lines are also kept in memory"""

    def read_data(self, paths, dictionary):
        for path in paths:
            with open(path, "r") as f:
                file_lines = []
                file_tokens_list = []
                file_sizes = []
                for line in f:
                    file_lines.append(line.strip("\n"))
                    tokens = dictionary.encode_line(
                        line,
                        add_if_not_exist=False,
                        append_eos=self.append_eos,
                        reverse_order=self.reverse_order,
                    )
                    file_tokens_list.append(tokens)
                    file_sizes.append(len(tokens))
                self.lines.append(file_lines)
                self.tokens_list.append(file_tokens_list)
                self.sizes.append(file_sizes)
        # Zip all sentences for each sample together
        self.lines = list(zip(*self.lines))
        self.tokens_list = list(zip(*self.tokens_list))
        # Sum sentence sizes for each sample
        self.sizes = np.asarray(self.sizes).sum(axis=0)
