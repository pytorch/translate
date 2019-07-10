#!/usr/bin/env python3

from fairseq import models
from fairseq.data import FairseqDataset, data_utils
from fairseq.models import FairseqMultiModel
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from pytorch_translate import vocab_constants
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
    ):
        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        indices = data_utils.filter_by_size(
            indices,
            dataset.size,
            max_positions,
            raise_exception=(not ignore_invalid_inputs),
        )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices,
            dataset.num_tokens,
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

    def get_encoder_langtok(self, src_lang, tgt_lang):
        if self.args.encoder_langtok is not None:
            if self.args.encoder_langtok == "src":
                if src_lang in vocab_constants.DIALECT_CODES:
                    return vocab_constants.DIALECT_CODES[src_lang]
            else:
                if tgt_lang in vocab_constants.DIALECT_CODES:
                    return vocab_constants.DIALECT_CODES[tgt_lang]
        # if encoder_langtok is not None or src_lang and tgt_lang are not in
        # vocab_constants.DIALECT_CODES
        return self.dicts[src_lang].eos()

    def get_decoder_langtok(self, tgt_lang):
        if (
            not self.args.decoder_langtok
            or tgt_lang not in vocab_constants.DIALECT_CODES
        ):
            return self.dicts[tgt_lang].eos()
        else:
            return vocab_constants.DIALECT_CODES[tgt_lang]
