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

    TODO: Refactor this class to look like WeightedBacktranslationDataset.
    We could wrap an existing dataset object and provide additional weights
    feature. This way, it will be more composable and can be used with arbitrary
    datasets. See D13143051.

    Args:
        weights (list): list of per example weight values; each example
        has a weight in [0.0, 1.0]. Alternatively, when weights consists of a
        single value, that value is broadcast as weight to all examples. [0.0]
        gives 0 weight to all examples.
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
        self.src_dict = src_dict

    def __getitem__(self, i):
        example = super().__getitem__(i)
        if self.weights:
            """
            If weight for example is missing, use last seen weight. Sometimes we just
            want to assign a weight to the entire dataset with a single value but also
            maintain the list convention of weights. This way, even if we don't care/know
            about dataset size, we can assign same weight to all examples.
            """
            if len(self.weights) <= i:
                example["weight"] = self.weights[-1]
            else:
                example["weight"] = self.weights[i]
        else:
            example["weight"] = 1.0

        return example

    def __len__(self):
        return super().__len__()

    def collater(self, samples):
        return WeightedLanguagePairDataset.collate(
            samples, self.src_dict.pad(), self.src_dict.eos()
        )

    @staticmethod
    def collate(samples, pad_idx, eos_idx, left_pad_source=False):
        if len(samples) == 0:
            return {}
        unweighted_data = data.language_pair_dataset.collate(
            samples, pad_idx, eos_idx, left_pad_source
        )
        original_weights = torch.FloatTensor([s.get("weight", 1.0) for s in samples])
        # sort by descending source length
        src_lengths = torch.LongTensor([s["source"].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        weights = original_weights.index_select(0, sort_order)
        unweighted_data["weights"] = weights
        return unweighted_data


class WeightedBacktranslationDataset(
    data.backtranslation_dataset.BacktranslationDataset
):
    """
    Extension of fairseq.data.BacktranslationDataset where each example
    has a weight in [0.0, 1.0], which will be used to weigh the loss.

    Args:
        weights (list): list of per example weight values; each example
        has a weight in [0.0, 1.0]. Alternatively, when weights consists of a
        single value, that value is broadcast as weight to all examples. [0.0]
        gives 0 weight to all examples.
    """

    def __init__(self, dataset, weights=None, **kwargs):
        self.weights = weights
        self.dataset = dataset

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.dataset, attr)

    def __getitem__(self, i):
        example = self.dataset.__getitem__(i)
        if self.weights:
            """
            If weight for example is missing, use last seen weight. Sometimes we just
            want to assign a weight to the entire dataset with a single value but also
            maintain the list convention of weights. This way, even if we don't care or
            don't know about dataset size, we can assign same weight to all examples.
            """
            if len(self.weights) <= i:
                example["weight"] = self.weights[-1]
            else:
                example["weight"] = self.weights[i]
        else:
            example["weight"] = 1.0

        return example

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        unweighted_data = self.dataset.collater(samples)
        original_weights = torch.FloatTensor([s.get("weight", 1.0) for s in samples])
        # sort by descending source length
        src_lengths = torch.LongTensor([s["source"].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        weights = original_weights.index_select(0, sort_order)
        unweighted_data["weights"] = weights
        return unweighted_data
