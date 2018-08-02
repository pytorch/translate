#!/usr/bin/env python3

import os
import unittest

import numpy as np
import torch
from fairseq.trainer import Trainer
from pytorch_translate import rnn  # noqa
from pytorch_translate import tasks, train
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
        test_args = test_utils.ModelParamsDict()
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        encoder_embed_path, embed_weights = test_utils.create_pretrained_embed(
            src_dict, test_args.encoder_hidden_dim
        )
        test_args.encoder_embed_path = encoder_embed_path
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        model = task.build_model(test_args)
        assert np.allclose(
            model.encoder.embed_tokens.weight.data.numpy(),
            embed_weights,
        )
        os.remove(encoder_embed_path)

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

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_load_checkpoint(self):
        test_save_file = test_utils.make_temp_file()
        test_args = test_utils.ModelParamsDict()
        test_args.distributed_rank = 0
        extra_state = test_utils.create_dummy_extra_state(epoch=2)
        trainer, _ = self._gpu_train_step(test_args)
        trainer.save_checkpoint(test_save_file, extra_state)
        loaded, extra_state = train.load_existing_checkpoint(
            test_save_file, trainer, restore_state=True
        )
        # Loading checkpoint without restore state should reset extra state
        assert loaded and extra_state["epoch"] == 2
        os.remove(test_save_file)

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_load_checkpoint_no_restore_state(self):
        test_save_file = test_utils.make_temp_file()
        test_args = test_utils.ModelParamsDict()
        test_args.distributed_rank = 0
        extra_state = test_utils.create_dummy_extra_state(epoch=2)
        trainer, _ = self._gpu_train_step(test_args)
        trainer.save_checkpoint(test_save_file, extra_state)
        loaded, extra_state = train.load_existing_checkpoint(
            test_save_file, trainer, restore_state=False
        )
        # Loading checkpoint without restore state should reset extra state
        assert loaded and extra_state is None
        os.remove(test_save_file)
