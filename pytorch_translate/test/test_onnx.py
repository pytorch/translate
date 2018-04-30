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
    CombinedDecoderEnsemble,
    CombinedEncoderEnsemble,
    DecoderBatchedStepEnsemble,
    DecoderStepEnsemble,
    EncoderEnsemble,
)
from pytorch_translate.test import utils as test_utils

from caffe2.python.onnx import backend as caffe2_backend


logger = logging.getLogger(__name__)


class TestONNX(unittest.TestCase):

    class BeamSearch(torch.jit.ScriptModule):

        def __init__(self, test_args, src_dict, tgt_dict):
            super().__init__(False)
            # Create models
            self.num_models = 3
            self.model_list = []

            self.incremental_states = [[{} for _ in range(len(self.model_list))]]
            for _ in range(self.num_models):
                self.model_list.append(models.build_model(
                    test_args, src_dict, tgt_dict))
            self.encoder_ensemble = CombinedEncoderEnsemble(self.model_list)
            self.decoder_ensemble = CombinedDecoderEnsemble(self.model_list)

            def encoder(src_tokens, src_lengths):
                self.incremental_states[0] = [{} for _ in range(len(self.model_list))]
                enc_outs = self.encoder_ensemble(
                    src_tokens.repeat(6, 1), src_lengths.repeat(6))
                return enc_outs

            def decoder(input_tokens, encoder_outs, final_hidden, final_cell,
                        src_lengths, src_tokens):
                dec_outs = self.decoder_ensemble(
                    input_tokens, encoder_outs, final_hidden, final_cell,
                    src_lengths, src_tokens, self.incremental_states[0])
                return dec_outs

            def reorder_states(new_order):
                for i, model in enumerate(self.decoder_ensemble.models):
                    model.decoder.reorder_incremental_state(
                        self.incremental_states[0][i], new_order.squeeze(dim=0))
                return torch.zeros(1, 2, 3)  # prevent DCE

            self.define('''
            def forward(
                self, src_tokens, src_lengths, scores_t, hypo_t, tokens_t, attention_t
            ):
                encoder_outs, final_hidden, final_cells, src_lengths, src_tokens\
                 = encoder(src_tokens, src_lengths)

                output_token_beam_list = tokens_t
                output_prev_index_beam_list = hypo_t
                output_score_beam_list = scores_t
                output_attention_weights_beam_list = attention_t
                dce = tokens_t

                timestep = 0

                while timestep < 20:
                    log_probs, attn = decoder(output_token_beam_list,
                        encoder_outs, final_hidden, final_cells, src_lengths,
                        src_tokens)

                    best_scores_per_hypo, best_tokens_per_hypo = topk(
                        log_probs, k=6, dim=-1, largest=1, sorted=1)

                    output_scores = best_scores_per_hypo + \
                        squeeze(scores_t, dim=0)

                    output_scores_flattened = view(output_scores, size=[-1])
                    output_scores_flattened_slice = output_scores_flattened
                    if timestep == 0:
                        output_scores_flattened_slice = slice(
                            output_scores_flattened, dim=0, end=6, start=0,
                            step=1)
                    else:
                        output_scores_flattened_slice = slice(
                            output_scores_flattened, dim=0, end=-1, start=0,
                            step=1)
                    output_scores_flattened_slice = view(
                        output_scores_flattened_slice, size=[1, -1])

                    scores_t, best_indices = topk(
                        output_scores_flattened_slice, k=6, dim=-1,
                        largest=1, sorted=1)

                    hypo_t_int64 = best_indices / 6

                    attention_t = index_select(
                        attention_t, squeeze(hypo_t_int64, dim=0), dim=0)
                    tokens_t_int64 = view(best_tokens_per_hypo, size=[-1])
                    tokens_t_int64 = index_select(
                        tokens_t_int64, squeeze(best_indices, dim=0), dim=0)
                    tokens_t_int64 = view(tokens_t_int64, size=[-1, 1])

                    output_token_beam_list = cat(
                        output_token_beam_list, tokens_t_int64, dim=1)
                    output_prev_index_beam_list = cat(
                        output_prev_index_beam_list, hypo_t_int64, dim=0)
                    output_score_beam_list = cat(
                        output_score_beam_list, scores_t, dim=0)
                    output_attention_weights_beam_list = cat(
                        output_attention_weights_beam_list, attention_t, dim=0)

                    timestep += 1

                    dce = reorder_states(hypo_t_int64)

                return output_token_beam_list, output_prev_index_beam_list, \
                    output_score_beam_list, output_attention_weights_beam_list,\
                    dce
            ''')


    def test_beam_search(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True,
            sequence_lstm=True,
        )

        # Create data
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        sample = next(samples)
        # TODO batched beam search
        src_tokens = sample['net_input']['src_tokens'][0]
        src_tokens = torch.unsqueeze(src_tokens, dim=0)
        src_lengths = sample['net_input']['src_lengths'][0]
        src_lengths = torch.unsqueeze(src_lengths, dim=0)

        bs = TestONNX.BeamSearch(test_args, src_dict, tgt_dict)

        scores_t = torch.zeros(1, 6)
        hypo_t = torch.zeros(1, 6).type(torch.LongTensor)
        eos = bs.model_list[0].dst_dict.eos()
        tokens_t = (torch.ones(6, 1) * eos).type(torch.LongTensor)
        attention_t = torch.nn.functional.softmax(torch.randn(6, 20), dim=-1)

        bs.forward(src_tokens, src_lengths, scores_t, hypo_t, tokens_t, attention_t)

    def _test_ensemble_encoder_export(self, test_args):
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)

        num_models = 3
        model_list = []
        for _ in range(num_models):
            model_list.append(models.build_model(test_args, src_dict, tgt_dict))
        encoder_ensemble = EncoderEnsemble(model_list)

        tmp_dir = tempfile.mkdtemp()
        encoder_pb_path = os.path.join(tmp_dir, 'encoder.pb')
        encoder_ensemble.onnx_export(encoder_pb_path)

        # test equivalence
        # The discrepancy in types here is a temporary expedient.
        # PyTorch indexing requires int64 while support for tracing
        # pack_padded_sequence() requires int32.
        sample = next(samples)
        src_tokens = sample['net_input']['src_tokens'][0:1].t()
        src_lengths = sample['net_input']['src_lengths'][0:1].int()

        pytorch_encoder_outputs = encoder_ensemble(src_tokens, src_lengths)

        with open(encoder_pb_path, 'r+b') as f:
            onnx_model = onnx.load(f)
        onnx_encoder = caffe2_backend.prepare(onnx_model)

        caffe2_encoder_outputs = onnx_encoder.run(
            (
                src_tokens.numpy(),
                src_lengths.numpy(),
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

    def test_ensemble_encoder_export_default(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True,
            sequence_lstm=True,
        )
        self._test_ensemble_encoder_export(test_args)

    def test_ensemble_encoder_export_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True,
            sequence_lstm=True,
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            'lexical_dictionaries': lexical_dictionaries,
            'num_top_words': 5,
            'max_translation_candidates_per_word': 1,
        }

        self._test_ensemble_encoder_export(test_args)

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

    def test_full_ensemble_export_default(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True,
            sequence_lstm=True,
        )
        self._test_full_ensemble_export(test_args)

    def test_full_ensemble_export_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True,
            sequence_lstm=True,
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            'lexical_dictionaries': lexical_dictionaries,
            'num_top_words': 5,
            'max_translation_candidates_per_word': 1,
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
        src_tokens = sample['net_input']['src_tokens'][0:1].t()
        src_lengths = sample['net_input']['src_lengths'][0:1].int()

        pytorch_encoder_outputs = encoder_ensemble(src_tokens, src_lengths)

        decoder_step_ensemble = DecoderBatchedStepEnsemble(
            model_list,
            beam_size=beam_size,
        )

        tmp_dir = tempfile.mkdtemp()
        decoder_step_pb_path = os.path.join(tmp_dir, 'decoder_step.pb')
        decoder_step_ensemble.onnx_export(
            decoder_step_pb_path,
            pytorch_encoder_outputs,
        )

        # single EOS in flat array
        input_tokens = torch.LongTensor(
            np.array([model_list[0].dst_dict.eos()]),
        )
        prev_scores = torch.FloatTensor(np.array([0.0]))
        timestep = torch.LongTensor(np.array([0]))

        pytorch_first_step_outputs = decoder_step_ensemble(
            input_tokens,
            prev_scores,
            timestep,
            *pytorch_encoder_outputs
        )

        # next step inputs (input_tokesn shape: [beam_size])
        next_input_tokens = torch.LongTensor(
            np.array([i for i in range(4, 9)]),
        )

        next_prev_scores = pytorch_first_step_outputs[1]
        next_timestep = timestep + 1
        next_states = pytorch_first_step_outputs[4:]

        step_inputs = []

        # encoder outputs need to be replicated for each input hypothesis
        for encoder_rep in pytorch_encoder_outputs[:len(model_list)]:
            step_inputs.append(encoder_rep.repeat(1, beam_size, 1))

        if model_list[0].decoder.vocab_reduction_module is not None:
            step_inputs.append(pytorch_encoder_outputs[len(model_list)])

        step_inputs.extend(list(next_states))

        pytorch_next_step_outputs = decoder_step_ensemble(
            next_input_tokens,
            next_prev_scores,
            next_timestep,
            *step_inputs
        )

        with open(decoder_step_pb_path, 'r+b') as f:
            onnx_model = onnx.load(f)
        onnx_decoder = caffe2_backend.prepare(onnx_model)

        decoder_inputs_numpy = [
            next_input_tokens.numpy(),
            next_prev_scores.detach().numpy(),
            next_timestep.detach().numpy(),
        ]
        for tensor in step_inputs:
            decoder_inputs_numpy.append(tensor.detach().numpy())

        caffe2_next_step_outputs = onnx_decoder.run(
            tuple(decoder_inputs_numpy),
        )

        for i in range(len(pytorch_next_step_outputs)):
            caffe2_out_value = caffe2_next_step_outputs[i]
            pytorch_out_value = pytorch_next_step_outputs[i].data.numpy()
            np.testing.assert_allclose(
                caffe2_out_value,
                pytorch_out_value,
                rtol=1e-4,
                atol=1e-6,
            )

    def test_batched_beam_decoder_default(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True,
            sequence_lstm=True,
        )
        self._test_batched_beam_decoder_step(test_args)

    def test_batched_beam_decoder_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True,
            sequence_lstm=True,
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            'lexical_dictionaries': lexical_dictionaries,
            'num_top_words': 5,
            'max_translation_candidates_per_word': 1,
        }
        self._test_batched_beam_decoder_step(test_args)
