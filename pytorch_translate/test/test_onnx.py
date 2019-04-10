#!/usr/bin/env python3

import logging
import os
import tempfile
import unittest

import numpy as np
import onnx
import torch
from caffe2.python.onnx import backend as caffe2_backend
from fairseq import models
from pytorch_translate import char_source_hybrid  # noqa
from pytorch_translate import char_source_model  # noqa
from pytorch_translate import char_source_transformer_model  # noqa
from pytorch_translate import rnn  # noqa
from pytorch_translate import transformer  # noqa
from pytorch_translate.ensemble_export import (
    BeamSearch,
    CharSourceEncoderEnsemble,
    DecoderBatchedStepEnsemble,
    EncoderEnsemble,
    ForcedDecoder,
    merge_transpose_and_batchmatmul,
)
from pytorch_translate.research.knowledge_distillation import (  # noqa
    dual_decoder_kd_model,
    hybrid_dual_decoder_kd_model,
)
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


logger = logging.getLogger(__name__)


class TestONNX(unittest.TestCase):
    def _test_ensemble_encoder_export(self, test_args):
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)

        num_models = 3
        model_list = []
        for _ in range(num_models):
            model_list.append(task.build_model(test_args))
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

        onnx_encoder = caffe2_backend.prepare_zip_archive(encoder_pb_path)

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

    def test_ensemble_transformer_encoder_export(self):
        test_args = test_utils.ModelParamsDict(arch="transformer")
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

        onnx_encoder = caffe2_backend.prepare_zip_archive(encoder_pb_path)

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

    def _test_batched_beam_decoder_step(self, test_args, return_caffe2_rep=False):
        beam_size = 5
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)

        num_models = 3
        model_list = []
        for _ in range(num_models):
            model_list.append(task.build_model(test_args))
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
            model_list, tgt_dict, beam_size=beam_size
        )

        tmp_dir = tempfile.mkdtemp()
        decoder_step_pb_path = os.path.join(tmp_dir, "decoder_step.pb")
        decoder_step_ensemble.onnx_export(decoder_step_pb_path, pytorch_encoder_outputs)

        # single EOS in flat array
        input_tokens = torch.LongTensor(np.array([tgt_dict.eos()]))
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

        onnx_decoder = caffe2_backend.prepare_zip_archive(decoder_step_pb_path)

        if return_caffe2_rep:
            return onnx_decoder

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
        decoder_step_ensemble.save_to_db(
            output_path=os.path.join(tmp_dir, "decoder.predictor_export"),
            encoder_ensemble_outputs=pytorch_encoder_outputs,
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

    def test_batched_beam_decoder_transformer(self):
        test_args = test_utils.ModelParamsDict(arch="transformer")
        self._test_batched_beam_decoder_step(test_args)

    def test_batched_beam_decoder_transformer_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(arch="transformer")
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 5,
            "max_translation_candidates_per_word": 1,
        }
        self._test_batched_beam_decoder_step(test_args)

    def test_batched_beam_decoder_hybrid_transformer_rnn(self):
        test_args = test_utils.ModelParamsDict(arch="hybrid_transformer_rnn")
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 5,
            "max_translation_candidates_per_word": 1,
        }
        self._test_batched_beam_decoder_step(test_args)

    def _test_forced_decoder_export(self, test_args):
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)

        num_models = 3
        model_list = []
        for _ in range(num_models):
            model_list.append(task.build_model(test_args))

        forced_decoder_ensemble = ForcedDecoder(
            model_list, tgt_dict, word_reward=0.25, unk_reward=-0.5
        )

        tmp_dir = tempfile.mkdtemp()
        forced_decoder_pb_path = os.path.join(tmp_dir, "forced_decoder.pb")
        forced_decoder_ensemble.onnx_export(forced_decoder_pb_path)

    def test_forced_decoder_export_default(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        self._test_forced_decoder_export(test_args)

    def test_forced_decoder_export_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 5,
            "max_translation_candidates_per_word": 1,
        }
        self._test_forced_decoder_export(test_args)

    def _test_ensemble_encoder_export_char_source(self, test_args):
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)

        num_models = 3
        model_list = []
        for _ in range(num_models):
            model_list.append(task.build_model(test_args))
        encoder_ensemble = CharSourceEncoderEnsemble(model_list)

        tmp_dir = tempfile.mkdtemp()
        encoder_pb_path = os.path.join(tmp_dir, "char_encoder.pb")
        encoder_ensemble.onnx_export(encoder_pb_path)

        length = 5
        src_tokens = torch.LongTensor(np.ones((length, 1), dtype="int64"))
        src_lengths = torch.IntTensor(np.array([length], dtype="int32"))
        word_length = 3
        char_inds = torch.LongTensor(np.ones((1, length, word_length), dtype="int64"))
        word_lengths = torch.IntTensor(
            np.array([word_length] * length, dtype="int32")
        ).reshape((1, length))

        pytorch_encoder_outputs = encoder_ensemble(
            src_tokens, src_lengths, char_inds, word_lengths
        )

        onnx_encoder = caffe2_backend.prepare_zip_archive(encoder_pb_path)

        caffe2_encoder_outputs = onnx_encoder.run(
            (
                src_tokens.numpy(),
                src_lengths.numpy(),
                char_inds.numpy(),
                word_lengths.numpy(),
            )
        )

        for i in range(len(pytorch_encoder_outputs)):
            caffe2_out_value = caffe2_encoder_outputs[i]
            pytorch_out_value = pytorch_encoder_outputs[i].detach().numpy()
            np.testing.assert_allclose(
                caffe2_out_value, pytorch_out_value, rtol=1e-4, atol=1e-6
            )

        encoder_ensemble.save_to_db(os.path.join(tmp_dir, "encoder.predictor_export"))

    def test_ensemble_encoder_export_char_cnn_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 5,
            "max_translation_candidates_per_word": 1,
        }

        test_args.arch = "char_source"
        test_args.char_source_dict_size = 126
        test_args.char_embed_dim = 8
        test_args.char_cnn_params = "[(10, 3), (10, 5)]"
        test_args.char_cnn_nonlinear_fn = "tanh"
        test_args.char_cnn_pool_type = "max"
        test_args.char_cnn_num_highway_layers = 2

        self._test_ensemble_encoder_export_char_source(test_args)

    def test_ensemble_encoder_export_unk_only_char_cnn_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 5,
            "max_translation_candidates_per_word": 1,
        }

        test_args.arch = "char_source"
        test_args.char_source_dict_size = 126
        test_args.char_embed_dim = 8
        test_args.char_cnn_params = "[(50, 1), (76, 2), (130, 3)]"
        test_args.char_cnn_nonlinear_fn = "relu"
        test_args.char_cnn_pool_type = "max"
        test_args.char_cnn_num_highway_layers = 2
        test_args.char_cnn_output_dim = 64
        test_args.encoder_embed_dim = 64
        test_args.unk_only_char_encoding = True

        self._test_ensemble_encoder_export_char_source(test_args)

    def test_ensemble_encoder_export_char_rnn_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 5,
            "max_translation_candidates_per_word": 1,
        }

        test_args.arch = "char_source"
        test_args.char_source_dict_size = 126
        test_args.char_embed_dim = 8
        test_args.char_rnn_units = 12
        test_args.char_rnn_layers = 2

        self._test_ensemble_encoder_export_char_source(test_args)

    def test_ensemble_encoder_export_char_cnn_hybrid(self):
        test_args = test_utils.ModelParamsDict(arch="hybrid_transformer_rnn")

        test_args.arch = "char_source_hybrid"
        test_args.char_source_dict_size = 126
        test_args.char_embed_dim = 8
        test_args.char_cnn_params = "[(10, 3), (10, 5)]"
        test_args.char_cnn_nonlinear_fn = "tanh"
        test_args.char_cnn_pool_type = "max"
        test_args.char_cnn_num_highway_layers = 2

        self._test_ensemble_encoder_export_char_source(test_args)

    def test_char_rnn_equivalent(self):
        """Ensure that the CharRNNEncoder.onnx_export_model path does not
        change computation"""
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 5,
            "max_translation_candidates_per_word": 1,
        }

        test_args.arch = "char_source"
        test_args.char_source_dict_size = 126
        test_args.char_embed_dim = 8
        test_args.char_rnn_units = 12
        test_args.char_rnn_layers = 2

        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)

        num_models = 3
        model_list = []
        for _ in range(num_models):
            model_list.append(task.build_model(test_args))
        encoder_ensemble = CharSourceEncoderEnsemble(model_list)

        length = 5
        src_tokens = torch.LongTensor(
            np.random.randint(0, len(src_dict), (length, 1), dtype="int64")
        )
        src_lengths = torch.IntTensor(np.array([length], dtype="int32"))
        word_length = 3
        char_inds = torch.LongTensor(
            np.random.randint(0, 126, (1, length, word_length), dtype="int64")
        )
        word_lengths = torch.IntTensor(
            np.array([word_length] * length, dtype="int32")
        ).reshape((1, length))

        onnx_path_outputs = encoder_ensemble(
            src_tokens, src_lengths, char_inds, word_lengths
        )

        for model in encoder_ensemble.models:
            model.encoder.onnx_export_model = False

        original_path_outputs = encoder_ensemble(
            src_tokens, src_lengths, char_inds, word_lengths
        )

        for (onnx_out, original_out) in zip(onnx_path_outputs, original_path_outputs):
            onnx_array = onnx_out.detach().numpy()
            original_array = original_out.detach().numpy()
            assert onnx_array.shape == original_array.shape
            np.testing.assert_allclose(onnx_array, original_array)

    def test_ensemble_char_transformer_encoder_export(self):
        test_args = test_utils.ModelParamsDict(arch="transformer")
        test_args.arch = "char_source_transformer"
        test_args.char_source_dict_size = 126
        test_args.char_embed_dim = 8
        test_args.char_cnn_params = "[(50, 1), (100,2)]"
        test_args.char_cnn_nonlinear_fn = "relu"
        test_args.char_cnn_num_highway_layers = 2
        test_args.char_cnn_pool_type = "max"
        self._test_ensemble_encoder_export_char_source(test_args)

    def test_merge_transpose_and_batchmatmul(self):
        test_args = test_utils.ModelParamsDict(arch="transformer")
        caffe2_rep = self._test_batched_beam_decoder_step(
            test_args, return_caffe2_rep=True
        )
        merge_transpose_and_batchmatmul(caffe2_rep)

    def test_ensemble_encoder_export_dual_decoder(self):
        test_args = test_utils.ModelParamsDict(arch="dual_decoder_kd")
        self._test_ensemble_encoder_export(test_args)

    def test_batched_beam_decoder_hybrid_reduced_attention(self):
        test_args = test_utils.ModelParamsDict(arch="hybrid_transformer_rnn")
        test_args.decoder_reduced_attention_dim = 10
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 5,
            "max_translation_candidates_per_word": 1,
        }
        self._test_batched_beam_decoder_step(test_args)

    def test_batched_beam_decoder_hybrid_dual_decoder_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(arch="hybrid_dual_decoder_kd")
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 5,
            "max_translation_candidates_per_word": 1,
        }
        self._test_batched_beam_decoder_step(test_args)
