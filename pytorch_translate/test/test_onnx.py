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


class TestONNX(unittest.TestCase):
    def _test_ensemble_encoder_export(self, test_args):
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)

        num_models = 3
        model_list = []
        for _ in range(num_models):
            model_list.append(models.build_model(test_args, src_dict, tgt_dict))
        encoder_ensemble = EncoderEnsemble(model_list)

        tmp_dir = tempfile.mkdtemp()
        encoder_pb_path = os.path.join(tmp_dir, "encoder.pb")
        encoder_ensemble.onnx_export(encoder_pb_path)

        # test equivalence
        # The discrepancy in types here is a temporary expedient.
        # PyTorch indexing requires int64 while support for tracing
        # pack_padded_sequence() requires int32.
        sample = next(samples)
        src_tokens = sample["net_input"]["src_tokens"][0:1].t()
        src_lengths = sample["net_input"]["src_lengths"][0:1].int()

        pytorch_encoder_outputs = encoder_ensemble(src_tokens, src_lengths)

        with open(encoder_pb_path, "r+b") as f:
            onnx_model = onnx.load(f)
        onnx_encoder = caffe2_backend.prepare(onnx_model)

        caffe2_encoder_outputs = onnx_encoder.run(
            (src_tokens.numpy(), src_lengths.numpy())
        )

        for i in range(len(pytorch_encoder_outputs)):
            caffe2_out_value = caffe2_encoder_outputs[i]
            pytorch_out_value = pytorch_encoder_outputs[i].detach().numpy()
            np.testing.assert_allclose(
                caffe2_out_value, pytorch_out_value, rtol=1e-4, atol=1e-6
            )

        encoder_ensemble.save_to_db(os.path.join(tmp_dir, "encoder.predictor_export"))

    def test_ensemble_encoder_export_default(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        self._test_ensemble_encoder_export(test_args)

    def test_ensemble_encoder_export_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 5,
            "max_translation_candidates_per_word": 1,
        }

        self._test_ensemble_encoder_export(test_args)

    def _test_ensemble_encoder_object_export(self, encoder_ensemble):
        tmp_dir = tempfile.mkdtemp()
        encoder_pb_path = os.path.join(tmp_dir, "encoder.pb")
        encoder_ensemble.onnx_export(encoder_pb_path)

        src_dict = encoder_ensemble.models[0].src_dict
        token_list = [src_dict.unk()] * 4 + [src_dict.eos()]
        src_tokens = torch.LongTensor(
            np.array(token_list, dtype="int64").reshape(-1, 1)
        )
        src_lengths = torch.IntTensor(np.array([len(token_list)], dtype="int32"))

        pytorch_encoder_outputs = encoder_ensemble(src_tokens, src_lengths)

        with open(encoder_pb_path, "r+b") as f:
            onnx_model = onnx.load(f)
        onnx_encoder = caffe2_backend.prepare(onnx_model)

        srclen = src_tokens.size(1)
        beam_size = 1

        src_tokens = src_tokens.repeat(1, beam_size).view(-1, srclen).numpy()
        src_lengths = src_lengths.repeat(beam_size).numpy()

        caffe2_encoder_outputs = onnx_encoder.run((src_tokens, src_lengths))

        for i in range(len(pytorch_encoder_outputs)):
            caffe2_out_value = caffe2_encoder_outputs[i]
            pytorch_out_value = pytorch_encoder_outputs[i].detach().numpy()
            np.testing.assert_allclose(
                caffe2_out_value, pytorch_out_value, rtol=1e-4, atol=1e-6
            )

        encoder_ensemble.save_to_db(os.path.join(tmp_dir, "encoder.predictor_export"))

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
        src_tokens = sample["net_input"]["src_tokens"][0:1].t()
        src_lengths = sample["net_input"]["src_lengths"][0:1].int()

        pytorch_encoder_outputs = encoder_ensemble(src_tokens, src_lengths)

        decoder_step_ensemble = DecoderStepEnsemble(model_list, beam_size=5)

        tmp_dir = tempfile.mkdtemp()
        decoder_step_pb_path = os.path.join(tmp_dir, "decoder_step.pb")
        decoder_step_ensemble.onnx_export(decoder_step_pb_path, pytorch_encoder_outputs)

        # single EOS
        input_token = torch.LongTensor(np.array([[model_list[0].dst_dict.eos()]]))
        timestep = torch.LongTensor(np.array([[0]]))

        pytorch_decoder_outputs = decoder_step_ensemble(
            input_token, timestep, *pytorch_encoder_outputs
        )

        with open(decoder_step_pb_path, "r+b") as f:
            onnx_model = onnx.load(f)
        onnx_decoder = caffe2_backend.prepare(onnx_model)

        decoder_inputs_numpy = [input_token.numpy(), timestep.numpy()]
        for tensor in pytorch_encoder_outputs:
            decoder_inputs_numpy.append(tensor.detach().numpy())

        caffe2_decoder_outputs = onnx_decoder.run(tuple(decoder_inputs_numpy))

        for i in range(len(pytorch_decoder_outputs)):
            caffe2_out_value = caffe2_decoder_outputs[i]
            pytorch_out_value = pytorch_decoder_outputs[i].detach().numpy()
            np.testing.assert_allclose(
                caffe2_out_value, pytorch_out_value, rtol=1e-4, atol=1e-6
            )

        decoder_step_ensemble.save_to_db(
            os.path.join(tmp_dir, "decoder_step.predictor_export"),
            pytorch_encoder_outputs,
        )

    def test_full_ensemble_export_default(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        self._test_full_ensemble_export(test_args)

    def test_full_ensemble_export_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 5,
            "max_translation_candidates_per_word": 1,
        }

        self._test_full_ensemble_export(test_args)

    def _test_batched_beam_decoder_step(self, test_args):
        beam_size = 5
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
        src_tokens = sample["net_input"]["src_tokens"][0:1].t()
        src_lengths = sample["net_input"]["src_lengths"][0:1].int()

        pytorch_encoder_outputs = encoder_ensemble(src_tokens, src_lengths)

        decoder_step_ensemble = DecoderBatchedStepEnsemble(
            model_list, beam_size=beam_size
        )

        tmp_dir = tempfile.mkdtemp()
        decoder_step_pb_path = os.path.join(tmp_dir, "decoder_step.pb")
        decoder_step_ensemble.onnx_export(decoder_step_pb_path, pytorch_encoder_outputs)

        # single EOS in flat array
        input_tokens = torch.LongTensor(np.array([model_list[0].dst_dict.eos()]))
        prev_scores = torch.FloatTensor(np.array([0.0]))
        timestep = torch.LongTensor(np.array([0]))

        pytorch_first_step_outputs = decoder_step_ensemble(
            input_tokens, prev_scores, timestep, *pytorch_encoder_outputs
        )

        # next step inputs (input_tokesn shape: [beam_size])
        next_input_tokens = torch.LongTensor(np.array([i for i in range(4, 9)]))

        next_prev_scores = pytorch_first_step_outputs[1]
        next_timestep = timestep + 1
        next_states = list(pytorch_first_step_outputs[4:])

        # Tile these for the next timestep
        for i in range(len(model_list)):
            next_states[i] = next_states[i].repeat(1, beam_size, 1)

        pytorch_next_step_outputs = decoder_step_ensemble(
            next_input_tokens, next_prev_scores, next_timestep, *next_states
        )

        with open(decoder_step_pb_path, "r+b") as f:
            onnx_model = onnx.load(f)
        onnx_decoder = caffe2_backend.prepare(onnx_model)

        decoder_inputs_numpy = [
            next_input_tokens.numpy(),
            next_prev_scores.detach().numpy(),
            next_timestep.detach().numpy(),
        ]
        for tensor in next_states:
            decoder_inputs_numpy.append(tensor.detach().numpy())

        caffe2_next_step_outputs = onnx_decoder.run(tuple(decoder_inputs_numpy))

        for i in range(len(pytorch_next_step_outputs)):
            caffe2_out_value = caffe2_next_step_outputs[i]
            pytorch_out_value = pytorch_next_step_outputs[i].detach().numpy()
            np.testing.assert_allclose(
                caffe2_out_value, pytorch_out_value, rtol=1e-4, atol=1e-6
            )

    def test_batched_beam_decoder_default(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        self._test_batched_beam_decoder_step(test_args)

    def test_batched_beam_decoder_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 5,
            "max_translation_candidates_per_word": 1,
        }
        self._test_batched_beam_decoder_step(test_args)

    def _test_full_beam_decoder(self, test_args):
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        sample = next(samples)
        src_tokens = sample["net_input"]["src_tokens"][0:1].t()
        src_lengths = sample["net_input"]["src_lengths"][0:1].int()

        num_models = 3
        model_list = []
        for _ in range(num_models):
            model_list.append(models.build_model(test_args, src_dict, tgt_dict))

        bs = BeamSearch(model_list, src_tokens, src_lengths, beam_size=6)
        prev_token = torch.LongTensor([0])
        prev_scores = torch.FloatTensor([0.0])
        attn_weights = torch.zeros(11)
        prev_hypos_indices = torch.zeros(6, dtype=torch.int64)

        outs = bs(
            src_tokens,
            src_lengths,
            prev_token,
            prev_scores,
            attn_weights,
            prev_hypos_indices,
            torch.LongTensor([20]),
        )

        import io

        f = io.BytesIO()
        torch.onnx._export(
            bs,
            (
                src_tokens,
                src_lengths,
                prev_token,
                prev_scores,
                attn_weights,
                prev_hypos_indices,
                torch.LongTensor([20]),
            ),
            f,
            export_params=True,
            verbose=False,
            example_outputs=outs,
        )

        torch.onnx._export_to_pretty_string(
            bs,
            (
                src_tokens,
                src_lengths,
                prev_token,
                prev_scores,
                attn_weights,
                prev_hypos_indices,
                torch.LongTensor([20]),
            ),
            f,
            export_params=True,
            verbose=False,
            example_outputs=outs,
        )

        f.seek(0)
        import onnx

        onnx_model = onnx.load(f)
        c2_model = caffe2_backend.prepare(onnx_model)
        c2_model.run(
            (
                src_tokens.numpy(),
                src_lengths.numpy(),
                prev_token.numpy(),
                prev_scores.numpy(),
                attn_weights.numpy(),
                prev_hypos_indices.numpy(),
                np.array([20]),
            )
        )

    @unittest.skip("Probably needs updated PyTorch")
    def test_full_beam_decoder(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        self._test_full_beam_decoder(test_args)
