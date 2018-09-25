#!/usr/bin/env python3

import torch
from fairseq import data


class IndexedWeightsDataset(data.indexed_dataset.IndexedDataset):
    def __init__(self, path):
        self.values = []
        self.read_data(path)

    def read_data(self, path):
        with open(path, "r") as f:
            for line in f:
                self.values.append(float(line.strip("\n")))
            self.size = len(self.values)

    def __getitem__(self, i):
        self.check_index(i)
        return self.values[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size


class WeightedLanguagePairDataset(data.language_pair_dataset.LanguagePairDataset):
    """
    Extension of fairseq.data.LanguagePairDataset where each example
    has a weight in [0.0, 1.0], which will be used to weigh the loss.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        weights=None,
        **kwargs,
    ):
        super().__init__(src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict, **kwargs)
        self.weights = weights

    def __getitem__(self, i):
        example = super().__getitem__(i)
        if self.weights:
            example["weight"] = self.weights[i]
        else:
            example["weight"] = 1.0

        return example

    def __len__(self):
        return super().__len__()

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        unweighted_data = super().collater(samples)
        original_weights = torch.FloatTensor([s.get("weight", 1.0) for s in samples])
        # sort by descending source length
        src_lengths = torch.LongTensor([s["source"].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        weights = original_weights.index_select(0, sort_order)
        unweighted_data["weights"] = weights
        return unweighted_data
