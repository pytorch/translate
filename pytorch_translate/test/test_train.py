#!/usr/bin/env python3

import os
import unittest

import numpy as np
import torch
from pytorch_translate import rnn  # noqa
from pytorch_translate import train
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestRNNModel(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_gpu_train_step(self):
        test_args = test_utils.ModelParamsDict()
        trainer, _ = test_utils.gpu_train_step(test_args)
        assert trainer.get_meter("gnorm").avg > 0

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_gpu_freeze_embedding(self):
        test_args = test_utils.ModelParamsDict(
            encoder_freeze_embed=True, decoder_freeze_embed=True
        )
        test_utils.gpu_train_step(test_args)

    def test_load_pretrained_embedding(self):
        test_args = test_utils.ModelParamsDict()
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        encoder_embed_path, embed_weights = test_utils.create_pretrained_embed(
            src_dict, test_args.encoder_hidden_dim
        )
        test_args.encoder_pretrained_embed = encoder_embed_path
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        model = task.build_model(test_args)
        assert np.allclose(
            model.encoder.embed_tokens.weight.data.numpy(), embed_weights
        )
        os.remove(encoder_embed_path)

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_milstm_cell(self):
        test_args = test_utils.ModelParamsDict(cell_type="milstm")
        trainer, _ = test_utils.gpu_train_step(test_args)
        assert trainer.get_meter("gnorm").avg > 0

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_sequence_lstm_encoder(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        trainer, _ = test_utils.gpu_train_step(test_args)
        assert trainer.get_meter("gnorm").avg > 0

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_layer_norm_lstm_cell(self):
        test_args = test_utils.ModelParamsDict(cell_type="layer_norm_lstm")
        trainer, _ = test_utils.gpu_train_step(test_args)
        assert trainer.get_meter("gnorm").avg > 0

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_first_layer_multihead_attention_(self):
        test_args = test_utils.ModelParamsDict(
            attention_type="multihead", attention_heads=2, first_layer_attention=True
        )
        trainer, _ = test_utils.gpu_train_step(test_args)
        assert trainer.get_meter("gnorm").avg > 0
