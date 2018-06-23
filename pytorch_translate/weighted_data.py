#!/usr/bin/env python3

import torch

from fairseq import data, indexed_dataset


class IndexedWeightsDataset(indexed_dataset.IndexedDataset):
    def __init__(self, path):
        self.values = []
        self.read_data(path)

    def read_data(self, path):
        with open(path, 'r') as f:
            for line in f:
                self.values.append(float(line.strip('\n')))
            self.size = len(self.values)

    def __getitem__(self, i):
        self.check_index(i)
        return self.values[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size


class WeightedLanguagePairDataset(data.LanguagePairDataset):
    """
    Extension of fairseq.data.LanguagePairDataset where each example
    has a weight in [0.0, 1.0], which will be used to weigh the loss.
    """

    def __init__(self, src, dst, pad_idx, eos_idx, weights=None):
        super().__init__(src, dst, pad_idx, eos_idx)
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
        unweighted_data = super().collater(samples)
        original_weights = torch.FloatTensor([s['weight'] for s in samples])
        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        weights = original_weights.index_select(0, sort_order)
        unweighted_data["weights"] = weights
        return unweighted_data
