#!/usr/bin/env python3

import numpy as np
from fairseq.data.concat_dataset import ConcatDataset


class LanguagePairUpsamplingDataset(ConcatDataset):
    def __init__(self, datasets, sample_ratios=1):
        super(LanguagePairUpsamplingDataset, self).__init__(datasets, sample_ratios)
        if isinstance(sample_ratios, float):
            self.memoized_sizes = [self.size(idx) for idx in range(len(self))]
        else:
            self.memoized_sizes = np.concatenate(
                [
                    np.tile(ds.src_sizes, sr)
                    for ds, sr in zip(self.datasets, self.sample_ratios)
                ]
            )

    @property
    def sizes(self):
        return self.memoized_sizes
