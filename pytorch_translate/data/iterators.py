#!/usr/bin/env python3

from fairseq.data import RoundRobinZipDatasets, iterators
from pytorch_translate.data import weighted_data


class WeightedEpochBatchIterator(iterators.EpochBatchIterator):
    def __init__(
        self,
        dataset,
        collate_fn,
        batch_sampler,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        weights=None,
    ):
        """
        Extension of fairseq.iterators.EpochBatchIterator to use an additional
        weights structure. This weighs datasets as a function of epoch value.

        Args:
            dataset (~torch.utils.data.Dataset): dataset from which to load the data
            collate_fn (callable): merges a list of samples to form a mini-batch
            batch_sampler (~torch.utils.data.Sampler): an iterator over batches of
                indices
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            weights: is of the format [(epoch, {dataset: weight})]
        """
        super().__init__(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
        )
        self.weights = weights

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator. Default: ``True``
            fix_batches_to_gpus: ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching. Default:
                ``False``
        """
        if self.weights and isinstance(self.dataset, RoundRobinZipDatasets):
            """
            Set dataset weight based on schedule and current epoch
            """
            prev_scheduled_epochs = 0
            dataset_weights_map = None
            for schedule in self.weights:
                # schedule looks like (num_epochs, {dataset: weight})
                if self.epoch <= schedule[0] + prev_scheduled_epochs:
                    dataset_weights_map = schedule[1]
                    break
                prev_scheduled_epochs += schedule[0]
            # Use last weights map if weights map is not specified for the current epoch
            if dataset_weights_map is None:
                dataset_weights_map = self.weights[-1][1]
            for dataset_name in self.dataset.datasets:
                if dataset_name in dataset_weights_map:
                    assert isinstance(
                        self.dataset.datasets[dataset_name],
                        weighted_data.WeightedLanguagePairDataset,
                    ) or isinstance(
                        self.dataset.datasets[dataset_name],
                        weighted_data.WeightedBacktranslationDataset,
                    )
                    self.dataset.datasets[dataset_name].weights = [
                        dataset_weights_map[dataset_name]
                    ]
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self.epoch += 1
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle, fix_batches_to_gpus=fix_batches_to_gpus
            )
        return self._cur_epoch_itr
