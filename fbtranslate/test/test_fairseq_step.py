#!/usr/bin/env python3

import logging
import numpy as np
import torch
import unittest

from fairseq import criterions, models, optim, utils
from fbtranslate import rnn  # noqa
from fbtranslate.test import utils as test_utils

logger = logging.getLogger(__name__)


class CompareFrameworkTest(unittest.TestCase):

    def run_fairseq_train_step(self, model, test_args, tgt_dict, sample):
        sample = utils.make_variable(sample)
        criterion = criterions.build_criterion(test_args, test_args, tgt_dict)
        optimizer = optim.build_optimizer(test_args, model.parameters())

        model.train()
        optimizer.zero_grad()

        loss, sample_size, logging_output = criterion(model, sample)

        # update on average loss but report total loss
        loss.div(sample_size).backward()

        grad_norm = torch.nn.utils.clip_grad_norm(
            model.parameters(),
            test_args.clip_norm,
        )
        optimizer.step()

        return loss, grad_norm

    def test_fairseq_step_cpu(self):
        test_args = test_utils.ModelParamsDict()
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        model = models.build_model(test_args, src_dict, tgt_dict)
        loss, grad_norm = self.run_fairseq_train_step(
            model,
            test_args,
            tgt_dict,
            next(samples),
        )
        assert grad_norm > 0

    def test_bidirectional_lstm(self):
        test_args = test_utils.ModelParamsDict()
        test_args.encoder_bidirectional = True
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        model = models.build_model(test_args, src_dict, tgt_dict)
        loss, grad_norm = self.run_fairseq_train_step(
            model,
            test_args,
            tgt_dict,
            next(samples),
        )
        assert grad_norm > 0

    def framework_comparison(self, test_args, is_variable_seqlen=False):
        caffe2_model = (
            test_utils.build_caffe2_model(test_args.caffe2_param_dict())
        )

        samples, src_dict, tgt_dict = test_utils.prepare_inputs(
            test_args,
            is_variable_seqlen=is_variable_seqlen,
        )

        pytorch_model = models.build_model(test_args, src_dict, tgt_dict)

        caffe2_params_before = test_utils.convert_params_to_caffe2(
            pytorch_model,
            test_args,
        )
        test_utils.load_caffe2_model(caffe2_params_before)

        sample = next(samples)
        caffe2_inputs = test_utils.convert_batch_to_caffe2(sample)
        test_utils.compare_weights(caffe2_model, pytorch_model, test_args)

        # Caffe2 step
        caffe2_loss = test_utils.caffe2_training_step(caffe2_model, caffe2_inputs)

        # PyTorch step
        pytorch_loss, grad_norm = self.run_fairseq_train_step(
            pytorch_model,
            test_args,
            tgt_dict,
            sample,
        )
        logger.info('PyTorch Loss: {}, Caffe2 Loss: {}'.format(
            pytorch_loss.data[0],
            caffe2_loss,
        ))
        np.testing.assert_allclose(caffe2_loss, pytorch_loss.data[0], rtol=1e-4)

        test_utils.compare_weights(caffe2_model, pytorch_model, test_args)

    def test_framework_comparison_default(self):
        test_args = test_utils.ModelParamsDict()
        self.framework_comparison(test_args)

    def test_framework_comparison_default_variable_seqlen(self):
        test_args = test_utils.ModelParamsDict()
        self.framework_comparison(test_args, is_variable_seqlen=True)

    def test_framework_comparison_bidirectional(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True,
        )
        self.framework_comparison(test_args)

    def test_framework_comparison_freeze_embedding(self):
        test_args = test_utils.ModelParamsDict(
            encoder_freeze_embed=True,
            decoder_freeze_embed=True,
        )
        self.framework_comparison(test_args)

    def test_framework_comparison_unclipped(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True,
            clip_norm=1000.0,
        )
        self.framework_comparison(test_args)

    def test_framework_comparison_sequence_lstm(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True,
            sequence_lstm=True,
        )
        self.framework_comparison(test_args, is_variable_seqlen=True)
