#!/usr/bin/env python3

from fairseq import models
from fairseq.data import data_utils, FairseqDataset
from fairseq.models import FairseqMultiModel
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from pytorch_translate.data import iterators as ptt_iterators


class PyTorchTranslateMultiTask(MultilingualTranslationTask):
    def build_model(self, args):
        model = models.build_model(args, self)
        if not isinstance(model, FairseqMultiModel):
            raise ValueError(
                "PyTorchTranslateMultiTask requires a FairseqMultiModel architecture"
            )
        return model

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        indices = data_utils.filter_by_size(
            indices, dataset, max_positions, raise_exception=(not ignore_invalid_inputs)
        )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices,
            num_tokens_fn=dataset.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        return ptt_iterators.WeightedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            weights=self.loss_weights,
        )

    def max_positions(self):
        """Return None to allow model to dictate max sentence length allowed"""
        return None
