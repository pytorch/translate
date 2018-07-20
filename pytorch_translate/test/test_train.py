#!/usr/bin/env python3

import torch
import unittest
import numpy as np

from fairseq.trainer import Trainer
from pytorch_translate import rnn  # noqa
from pytorch_translate import tasks
from pytorch_translate.test import utils as test_utils


class TestRNNModel(unittest.TestCase):
    def _gpu_train_step(self, test_args):
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        model = task.build_model(test_args)
        criterion = task.build_criterion(test_args)
        trainer = Trainer(test_args, task, model, criterion)
        logging_dict = trainer.train_step(next(samples))
        return trainer, logging_dict

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_gpu_train_step(self):
        test_args = test_utils.ModelParamsDict()
        trainer, _ = self._gpu_train_step(test_args)
        assert trainer.get_meter("gnorm").avg > 0

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_gpu_freeze_embedding(self):
        test_args = test_utils.ModelParamsDict(
            encoder_freeze_embed=True, decoder_freeze_embed=True
        )
        self._gpu_train_step(test_args)

    def test_load_pretrained_embedding(self):
        encoder_embedding = open(test_utils.make_temp_file(), "wb")
        test_args = test_utils.ModelParamsDict(
            encoder_pretrained_embed=encoder_embedding.name,
        )
        # The vocabulary defaults to 103 in test_utils.prepare_inputs.
        embed_array = np.random.random((103, test_args.encoder_embed_dim))
        np.save(encoder_embedding, embed_array)
        encoder_embedding.close()
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        model = models.build_model(test_args, src_dict, tgt_dict)
        assert np.allclose(
            model.encoder.embed_tokens.weight.data.numpy(),
            embed_array,
        )

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_gpu_pretrained_embedding(self):
        encoder_embedding = open(test_utils.make_temp_file(), "wb")
        decoder_embedding = open(test_utils.make_temp_file(), "wb")
        test_args = test_utils.ModelParamsDict(
            encoder_pretrained_embed=encoder_embedding.name,
            decoder_pretrained_embed=decoder_embedding.name,
            encoder_freeze_embed=True,
            decoder_freeze_embed=True,
        )
        # The vocabulary defaults to 103 in test_utils.prepare_inputs.
        np.save(encoder_embedding, np.zeros((103, test_args.encoder_embed_dim)))
        encoder_embedding.close()
        np.save(decoder_embedding, np.zeros((103, test_args.decoder_embed_dim)))
        decoder_embedding.close()
        self._gpu_train_step(test_args)

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_milstm_cell(self):
        test_args = test_utils.ModelParamsDict(cell_type="milstm")
        trainer, _ = self._gpu_train_step(test_args)
        assert trainer.get_meter("gnorm").avg > 0

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_sequence_lstm_encoder(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        trainer, _ = self._gpu_train_step(test_args)
        assert trainer.get_meter("gnorm").avg > 0

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_layer_norm_lstm_cell(self):
        test_args = test_utils.ModelParamsDict(cell_type="layer_norm_lstm")
        trainer, _ = self._gpu_train_step(test_args)
        assert trainer.get_meter("gnorm").avg > 0
