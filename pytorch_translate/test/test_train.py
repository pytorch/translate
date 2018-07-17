#!/usr/bin/env python3

import torch
import unittest

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
