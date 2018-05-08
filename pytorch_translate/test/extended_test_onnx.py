#!/usr/bin/env python3

import logging
import numpy as np
import onnx
import os
import tempfile
import torch
import unittest

from fairseq import models
from pytorch_translate import rnn  # noqa
from pytorch_translate.ensemble_export import (
    DecoderBatchedStepEnsemble,
    DecoderStepEnsemble,
    EncoderEnsemble,
    BeamSearch,
)
from pytorch_translate.test import utils as test_utils

from caffe2.python.onnx import backend as caffe2_backend


logger = logging.getLogger(__name__)


class ExtendedTestONNX(unittest.TestCase):
    """
    These tests have high latency due to reading GFS,
    and are thus marked as extended.
    """

    def _test_ensemble_encoder_object_export(self, encoder_ensemble):
        tmp_dir = tempfile.mkdtemp()
        encoder_pb_path = os.path.join(tmp_dir, 'encoder.pb')
        encoder_ensemble.onnx_export(encoder_pb_path)

        src_dict = encoder_ensemble.models[0].src_dict
        token_list = [src_dict.unk()] * 4 + [src_dict.eos()]
        src_tokens = torch.LongTensor(
            np.array(token_list, dtype='int64').reshape(-1, 1),
        )
        src_lengths = torch.IntTensor(
            np.array([len(token_list)], dtype='int32'),
        )

        pytorch_encoder_outputs = encoder_ensemble(src_tokens, src_lengths)

        with open(encoder_pb_path, 'r+b') as f:
            onnx_model = onnx.load(f)
        onnx_encoder = caffe2_backend.prepare(onnx_model)

        srclen = src_tokens.size(1)
        beam_size = 1

        src_tokens = src_tokens.repeat(1, beam_size).view(-1, srclen).numpy()
        src_lengths = src_lengths.repeat(beam_size).numpy()

        caffe2_encoder_outputs = onnx_encoder.run(
            (
                src_tokens,
                src_lengths,
            ),
        )

        for i in range(len(pytorch_encoder_outputs)):
            caffe2_out_value = caffe2_encoder_outputs[i]
            pytorch_out_value = pytorch_encoder_outputs[i].data.numpy()
            np.testing.assert_allclose(
                caffe2_out_value,
                pytorch_out_value,
                rtol=1e-4,
                atol=1e-6,
            )

        encoder_ensemble.save_to_db(
            os.path.join(tmp_dir, 'encoder.predictor_export'),
        )

    def test_export_encoder_from_checkpoints_no_vr(self):
        check_dir = (
            '/mnt/gfsdataswarm-global/namespaces/search/language-technology-mt/'
            'nnmt_tmp/tl_XX-en_XX-pytorch-testing-no-vocab-reduction-2'
        )
        checkpoints = [
            'averaged_checkpoint_best_3.pt',
            'averaged_checkpoint_best_4.pt',
        ]
        checkpoint_filenames = [os.path.join(check_dir, f) for f in checkpoints]

        encoder_ensemble = EncoderEnsemble.build_from_checkpoints(
            checkpoint_filenames,
            os.path.join(check_dir, 'dictionary-tl.txt'),
            os.path.join(check_dir, 'dictionary-en.txt'),
        )

        self._test_ensemble_encoder_object_export(encoder_ensemble)

    def _test_full_ensemble_export(self, test_args):
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)

        num_models = 3
        model_list = []
        for _ in range(num_models):
            model_list.append(models.build_model(test_args, src_dict, tgt_dict))
        encoder_ensemble = EncoderEnsemble(model_list)

        # test equivalence
        # The discrepancy in types here is a temporary expedient.
        # PyTorch indexing requires int64 while support for tracing
        # pack_padded_sequence() requires int32.
        sample = next(samples)
        src_tokens = sample['net_input']['src_tokens'][0:1].t()
        src_lengths = sample['net_input']['src_lengths'][0:1].int()

        pytorch_encoder_outputs = encoder_ensemble(src_tokens, src_lengths)

        decoder_step_ensemble = DecoderStepEnsemble(
            model_list,
            beam_size=5,
        )

        tmp_dir = tempfile.mkdtemp()
        decoder_step_pb_path = os.path.join(tmp_dir, 'decoder_step.pb')
        decoder_step_ensemble.onnx_export(
            decoder_step_pb_path,
            pytorch_encoder_outputs,
        )

        # single EOS
        input_token = torch.LongTensor(
            np.array([[model_list[0].dst_dict.eos()]]),
        )
        timestep = torch.LongTensor(np.array([[0]]))

        pytorch_decoder_outputs = decoder_step_ensemble(
            input_token,
            timestep,
            *pytorch_encoder_outputs
        )

        with open(decoder_step_pb_path, 'r+b') as f:
            onnx_model = onnx.load(f)
        onnx_decoder = caffe2_backend.prepare(onnx_model)

        decoder_inputs_numpy = [input_token.numpy(), timestep.numpy()]
        for tensor in pytorch_encoder_outputs:
            decoder_inputs_numpy.append(tensor.detach().numpy())

        caffe2_decoder_outputs = onnx_decoder.run(tuple(decoder_inputs_numpy))

        for i in range(len(pytorch_decoder_outputs)):
            caffe2_out_value = caffe2_decoder_outputs[i]
            pytorch_out_value = pytorch_decoder_outputs[i].data.numpy()
            np.testing.assert_allclose(
                caffe2_out_value,
                pytorch_out_value,
                rtol=1e-4,
                atol=1e-6,
            )

        decoder_step_ensemble.save_to_db(
            os.path.join(tmp_dir, 'decoder_step.predictor_export'),
            pytorch_encoder_outputs,
        )

    def _test_full_ensemble(
        self,
        encoder_ensemble,
        decoder_step_ensemble,
        batched_beam=False,
    ):
        tmp_dir = tempfile.mkdtemp()
        encoder_filename = os.path.join(tmp_dir, 'encoder.pb')
        encoder_ensemble.onnx_export(encoder_filename)

        src_dict = encoder_ensemble.models[0].src_dict
        token_list = [src_dict.unk()] * 4 + [src_dict.eos()]
        src_tokens = torch.LongTensor(
            np.array(token_list, dtype='int64').reshape(-1, 1),
        )
        src_lengths = torch.IntTensor(
            np.array([len(token_list)], dtype='int32'),
        )

        pytorch_encoder_outputs = encoder_ensemble(src_tokens, src_lengths)

        decoder_step_filename = os.path.join(tmp_dir, 'decoder_step.pb')
        decoder_step_ensemble.onnx_export(
            decoder_step_filename,
            pytorch_encoder_outputs,
        )

        dst_dict = decoder_step_ensemble.models[0].dst_dict
        input_token = torch.LongTensor(
            np.array([[dst_dict.eos()]], dtype='int64'),
        )
        timestep = torch.LongTensor(np.array([[0]]))

        if batched_beam:
            prev_scores = torch.FloatTensor(np.array([0.0]))
            pytorch_decoder_outputs = decoder_step_ensemble(
                input_token.view(-1),
                prev_scores,
                timestep.view(-1),
                *pytorch_encoder_outputs
            )
        else:
            pytorch_decoder_outputs = decoder_step_ensemble(
                input_token,
                timestep,
                *pytorch_encoder_outputs
            )

        with open(decoder_step_filename, 'r+b') as f:
            onnx_model = onnx.load(f)
        onnx_decoder = caffe2_backend.prepare(onnx_model)

        if batched_beam:
            decoder_inputs_numpy = [
                input_token.numpy(),
                prev_scores.numpy(),
                timestep.numpy(),
            ]
        else:
            decoder_inputs_numpy = [input_token.numpy(), timestep.numpy()]
        for tensor in pytorch_encoder_outputs:
            decoder_inputs_numpy.append(tensor.detach().numpy())

        caffe2_decoder_outputs = onnx_decoder.run(tuple(decoder_inputs_numpy))

        for i in range(len(pytorch_decoder_outputs)):
            caffe2_out_value = caffe2_decoder_outputs[i]
            pytorch_out_value = pytorch_decoder_outputs[i].data.numpy()
            np.testing.assert_allclose(
                caffe2_out_value,
                pytorch_out_value,
                rtol=1e-4,
                atol=1e-6,
            )

        decoder_step_ensemble.save_to_db(
            os.path.join(tmp_dir, 'decoder_step.predictor_export'),
            pytorch_encoder_outputs,
        )

    def test_full_model_from_checkpoints_no_vr(self):
        check_dir = (
            '/mnt/gfsdataswarm-global/namespaces/search/language-technology-mt/'
            'nnmt_tmp/tl_XX-en_XX-pytorch-testing-no-vocab-reduction-2'
        )
        checkpoints = [
            'averaged_checkpoint_best_3.pt',
            'averaged_checkpoint_best_4.pt',
        ]
        checkpoint_filenames = [os.path.join(check_dir, f) for f in checkpoints]

        encoder_ensemble = EncoderEnsemble.build_from_checkpoints(
            checkpoint_filenames,
            os.path.join(check_dir, 'dictionary-tl.txt'),
            os.path.join(check_dir, 'dictionary-en.txt'),
        )

        decoder_step_ensemble = DecoderStepEnsemble.build_from_checkpoints(
            checkpoint_filenames,
            os.path.join(check_dir, 'dictionary-tl.txt'),
            os.path.join(check_dir, 'dictionary-en.txt'),
            beam_size=5,
        )

        self._test_full_ensemble(encoder_ensemble, decoder_step_ensemble)

    def test_export_encoder_from_checkpoints_vr(self):
        check_dir = (
            '/mnt/gfsdataswarm-global/namespaces/search/language-technology-mt/'
            'nnmt_tmp/tl_XX-en_XX-pytorch-256-dim-vocab-reduction'
        )
        checkpoints = [
            'averaged_checkpoint_best_3.pt',
            'averaged_checkpoint_best_4.pt',
            'averaged_checkpoint_best_5.pt',
        ]
        checkpoint_filenames = [os.path.join(check_dir, f) for f in checkpoints]

        encoder_ensemble = EncoderEnsemble.build_from_checkpoints(
            checkpoint_filenames,
            os.path.join(check_dir, 'dictionary-tl.txt'),
            os.path.join(check_dir, 'dictionary-en.txt'),
        )

        self._test_ensemble_encoder_object_export(encoder_ensemble)

    def test_full_model_from_checkpoints_vr(self):
        check_dir = (
            '/mnt/gfsdataswarm-global/namespaces/search/language-technology-mt/'
            'nnmt_tmp/tl_XX-en_XX-pytorch-256-dim-vocab-reduction'
        )
        checkpoints = [
            'averaged_checkpoint_best_3.pt',
            'averaged_checkpoint_best_4.pt',
            'averaged_checkpoint_best_5.pt',
        ]
        checkpoint_filenames = [os.path.join(check_dir, f) for f in checkpoints]

        encoder_ensemble = EncoderEnsemble.build_from_checkpoints(
            checkpoint_filenames,
            os.path.join(check_dir, 'dictionary-tl.txt'),
            os.path.join(check_dir, 'dictionary-en.txt'),
        )

        decoder_step_ensemble = DecoderStepEnsemble.build_from_checkpoints(
            checkpoint_filenames,
            os.path.join(check_dir, 'dictionary-tl.txt'),
            os.path.join(check_dir, 'dictionary-en.txt'),
            beam_size=5,
        )

        self._test_full_ensemble(encoder_ensemble, decoder_step_ensemble)

    def test_batched_beam_from_checkpoints_vr(self):
        check_dir = (
            '/mnt/gfsdataswarm-global/namespaces/search/language-technology-mt/'
            'nnmt_tmp/tl_XX-en_XX-pytorch-256-dim-vocab-reduction'
        )
        checkpoints = [
            'averaged_checkpoint_best_3.pt',
            'averaged_checkpoint_best_4.pt',
            'averaged_checkpoint_best_5.pt',
        ]
        checkpoint_filenames = [os.path.join(check_dir, f) for f in checkpoints]

        encoder_ensemble = EncoderEnsemble.build_from_checkpoints(
            checkpoint_filenames,
            os.path.join(check_dir, 'dictionary-tl.txt'),
            os.path.join(check_dir, 'dictionary-en.txt'),
        )

        decoder_step_ensemble = DecoderBatchedStepEnsemble.build_from_checkpoints(
            checkpoint_filenames,
            os.path.join(check_dir, 'dictionary-tl.txt'),
            os.path.join(check_dir, 'dictionary-en.txt'),
            beam_size=5,
        )

        self._test_full_ensemble(
            encoder_ensemble,
            decoder_step_ensemble,
            batched_beam=True,
        )

    def _test_beam_search(
        self,
        beam_search,
    ):
        tmp_dir = tempfile.mkdtemp()

        src_dict = beam_search.models[0].src_dict
        # token_list = [src_dict.unk()] * 9 + [src_dict.eos()]
        token_list = [*range(123, 123+4)] + [src_dict.eos()]
        src_tokens = torch.LongTensor(
            np.array(token_list, dtype='int64').reshape(-1, 1),
        )
        src_lengths = torch.IntTensor(
            np.array([len(token_list)], dtype='int32'),
        )

        dst_dict = beam_search.models[0].dst_dict

        beam_search_filename = os.path.join(tmp_dir, 'beam_search.pb')
        beam_search.onnx_export(
            beam_search_filename,
        )

        prev_token = torch.LongTensor([dst_dict.eos()])
        prev_scores = torch.FloatTensor([0.0])
        attn_weights = torch.zeros(len(token_list))
        prev_hypos_indices = torch.zeros(beam_search.beam_size, dtype=torch.int64)
        num_steps = torch.LongTensor([20])

        inputs = (src_tokens, src_lengths, prev_token, prev_scores,
                  attn_weights, prev_hypos_indices, num_steps)

        pytorch_outputs = beam_search(*inputs)

        onnx_decoder = caffe2_backend.prepare_zip_archive(beam_search_filename)

        inputs_numpy = []
        for tensor in inputs:
            inputs_numpy.append(tensor.detach().numpy())

        caffe2_decoder_outputs = onnx_decoder.run(tuple(inputs_numpy))

        for i in range(len(pytorch_outputs)):
            caffe2_out_value = caffe2_decoder_outputs[i]
            pytorch_out_value = pytorch_outputs[i].data.numpy()
            np.testing.assert_allclose(
                caffe2_out_value,
                pytorch_out_value,
                rtol=1e-4,
                atol=1e-6,
            )

        beam_search.save_to_db(
            os.path.join(tmp_dir, 'beam_search.predictor_export'),
        )

    def test_beam_search(self):
        check_dir = (
            '/mnt/gfsdataswarm-global/namespaces/search/language-technology-mt/'
            'nnmt_tmp/tl_XX-en_XX-pytorch-256-dim-vocab-reduction'
        )
        checkpoints = [
            'averaged_checkpoint_best_3.pt',
            'averaged_checkpoint_best_4.pt',
            'averaged_checkpoint_best_5.pt',
        ]
        checkpoint_filenames = [os.path.join(check_dir, f) for f in checkpoints]

        beam_search = BeamSearch.build_from_checkpoints(
            checkpoint_filenames=checkpoint_filenames,
            src_dict_filename=os.path.join(check_dir, 'dictionary-tl.txt'),
            dst_dict_filename=os.path.join(check_dir, 'dictionary-en.txt'),
            beam_size=6,
        )


        self._test_beam_search(beam_search)
