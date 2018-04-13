#!/usr/bin/env python3

import logging
import numpy as np
import onnx
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fbtranslate import dictionary, rnn  # noqa

from caffe2.caffe2.fb.predictor import predictor_exporter
from caffe2.python import core, dyndep, workspace
from caffe2.python.onnx import backend as caffe2_backend

dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/aten:aten_op")

logger = logging.getLogger(__name__)


class CombinedEncoderEnsemble(nn.Module):

    def __init__(
        self,
        models,
    ):
        super().__init__()
        self.models = models
        for i, model in enumerate(self.models):
            self._modules['model_{}'.format(i)] = model

    def forward(self, src_tokens, src_lengths):
        outputs = []
        for model in self.models:
            o = list(model.encoder(src_tokens, src_lengths))
            for i in range(len(o)):
                o[i] = torch.unsqueeze(o[i], dim=0)
            outputs.append(tuple(o))

        outputs = [x for x in zip(*outputs)]
        for i in range(len(outputs)):
            outputs[i] = torch.cat(outputs[i], dim=0)

        return tuple(outputs)


class CombinedDecoderEnsemble(nn.Module):

    def __init__(
        self,
        models,
    ):
        super().__init__()
        self.models = models
        for i, model in enumerate(self.models):
            self._modules['model_{}'.format(i)] = model

    def forward(self, input_tokens, encoder_outs, final_hidden, final_cell,
                src_lengths, src_tokens, incremental_states):
        outputs = []
        for i, model in enumerate(self.models):
            per_model_inputs = (encoder_outs[i], final_hidden[i], final_cell[i],
                src_lengths[i], src_tokens[i])
            o = list(model.decoder(input_tokens, per_model_inputs,
                incremental_states[i]))
            if o[2] is None:
                del o[2]
            logits = o[0].view(-1, o[0].size(-1))
            o[0] = F.softmax(logits, dim=-1).view_as(o[0])
            for j in range(len(o)):
                o[j] = o[j].unsqueeze(dim=0)
            outputs.append(o)

        outputs = [x for x in zip(*outputs)]
        # Average across models in ensemble
        for i in range(len(outputs)):
            outputs[i] = torch.cat(outputs[i], dim=0)
            outputs[i] = torch.mean(outputs[i], dim=0)
            if i == 0:
                outputs[i] = torch.log(outputs[i])

        return tuple(outputs)


def onnx_export_ensemble(
    module,
    output_path,
    input_tuple,
    input_names,
    output_names,
):
    # include parameters as inputs of exported graph
    for name, _ in module.named_parameters():
        input_names.append(name)

    with open(output_path, 'w+b') as netdef_file:
        torch.onnx._export(
            module,
            input_tuple,
            netdef_file,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
        )


def load_models_from_checkpoints(
    checkpoint_filenames,
    src_dict_filename,
    dst_dict_filename,
):
    src_dict = dictionary.Dictionary.load(src_dict_filename)
    dst_dict = dictionary.Dictionary.load(dst_dict_filename)
    models = []
    for filename in checkpoint_filenames:
        checkpoint_data = torch.load(filename, map_location='cpu')

        model = rnn.RNNModel.build_model(
            checkpoint_data['args'],
            src_dict,
            dst_dict,
        )
        model.load_state_dict(checkpoint_data['model'])
        models.append(model)

    return models


def save_caffe2_rep_to_db(
    caffe2_backend_rep,
    output_path,
    input_names,
    output_names,
    num_workers,
):
    # netdef external_input includes internally produced blobs
    actual_external_inputs = set()
    produced = set()
    for operator in caffe2_backend_rep.predict_net.op:
        for blob in operator.input:
            if blob not in produced:
                actual_external_inputs.add(blob)
        for blob in operator.output:
            produced.add(blob)
    for blob in output_names:
        if blob not in produced:
            actual_external_inputs.add(blob)

    param_names = [
        blob for blob in actual_external_inputs
        if blob not in input_names
    ]

    init_net = core.Net(caffe2_backend_rep.init_net)
    predict_net = core.Net(caffe2_backend_rep.predict_net)

    # predictor_exporter requires disjoint params, inputs and outputs
    for i, param in enumerate(param_names):
        if param in output_names:
            saved_name = param + '_PARAM'
            init_net.Copy(param, saved_name)
            predict_net.Copy(saved_name, param)
            param_names[i] = saved_name

    output_shapes = {}
    for blob in output_names:
        output_shapes[blob] = (0,)

    with caffe2_backend_rep.workspace:
        workspace.RunNetOnce(init_net)
        predictor_export_meta = predictor_exporter.PredictorExportMeta(
            predict_net=predict_net,
            parameters=param_names,
            inputs=input_names,
            outputs=output_names,
            shapes=output_shapes,
            net_type='dag',
            num_workers=num_workers,
        )
        predictor_exporter.save_to_db(
            db_type='log_file_db',
            db_destination=output_path,
            predictor_export_meta=predictor_export_meta,
        )
    logger.info('Caffe2 predictor net saved as: {}'.format(output_path))


class EncoderEnsemble(nn.Module):

    def __init__(
        self,
        models,
    ):
        super().__init__()
        self.models = models
        for i, model in enumerate(self.models):
            self._modules['model_{}'.format(i)] = model

    def forward(self, src_tokens, src_lengths):
        outputs = []
        output_names = []
        states = []

        # (seq_length, batch_size) for compatibility with Caffe2
        src_tokens_seq_first = src_tokens.t()

        for i, model in enumerate(self.models):
            # evaluation mode
            model.eval()

            encoder_out = model.encoder(src_tokens_seq_first, src_lengths)

            # evaluation mode
            model.eval()

            # "primary" encoder output (vector representations per source token)
            encoder_outputs = encoder_out[0]
            outputs.append(encoder_outputs)
            output_names.append('encoder_output_{}'.format(i))

            init_hiddens, init_cells = model.decoder._init_prev_states(
                encoder_out,
            )
            for h, c in zip(init_hiddens, init_cells):
                states.extend([h, c])
            states.append(model.decoder.initial_attn_context)

        # underlying assumption is each model has same vocab_reduction_module
        vocab_reduction_module = self.models[0].decoder.vocab_reduction_module
        if vocab_reduction_module is not None:
            possible_translation_tokens = vocab_reduction_module(
                src_tokens=src_tokens,
                decoder_input_tokens=None,
            )
            outputs.append(possible_translation_tokens)
            output_names.append('possible_translation_tokens')

        for i, state in enumerate(states):
            outputs.append(state)
            output_names.append('initial_state_{}'.format(i))

        self.output_names = output_names

        return tuple(outputs)

    def onnx_export(self, output_path):
        # The discrepancy in types here is a temporary expedient.
        # PyTorch indexing requires int64 while support for tracing
        # pack_padded_sequence() requires int32.
        length = 5
        src_tokens = torch.LongTensor(np.ones((length, 1), dtype='int64'))
        src_lengths = torch.IntTensor(np.array([length], dtype='int32'))

        # generate output names
        self.forward(src_tokens, src_lengths)

        onnx_export_ensemble(
            module=self,
            output_path=output_path,
            input_tuple=(src_tokens, src_lengths),
            input_names=['encoder_inputs', 'encoder_lengths'],
            output_names=self.output_names,
        )

    def save_to_db(self, output_path):
        """
        Save encapsulated encoder export file.
        """
        tmp_dir = tempfile.mkdtemp()
        tmp_file = os.path.join(tmp_dir, 'encoder.pb')
        self.onnx_export(tmp_file)

        with open(tmp_file, 'r+b') as f:
            onnx_model = onnx.load(f)
        onnx_encoder = caffe2_backend.prepare(onnx_model)

        save_caffe2_rep_to_db(
            caffe2_backend_rep=onnx_encoder,
            output_path=output_path,
            input_names=['encoder_inputs', 'encoder_lengths'],
            output_names=self.output_names,
            num_workers=2 * len(self.models),
        )

    @staticmethod
    def build_from_checkpoints(
        checkpoint_filenames,
        src_dict_filename,
        dst_dict_filename,
    ):
        models = load_models_from_checkpoints(
            checkpoint_filenames,
            src_dict_filename,
            dst_dict_filename,
        )
        return EncoderEnsemble(models)


class DecoderStepEnsemble(nn.Module):

    def __init__(
        self,
        models,
        beam_size=1,
        word_penalty=0,
        unk_penalty=0,
    ):
        super().__init__()
        self.models = models
        for i, model in enumerate(self.models):
            model.decoder.attention.src_length_masking = False
            self._modules['model_{}'.format(i)] = model

        self.beam_size = beam_size
        self.word_penalty = word_penalty
        self.unk_penalty = unk_penalty

        dst_dict = models[0].dst_dict
        vocab_size = len(dst_dict.indices)
        self.word_rewards = torch.FloatTensor(vocab_size).zero_()
        self.word_rewards[:] = word_penalty
        self.word_rewards[dst_dict.eos()] = 0
        self.word_rewards[dst_dict.unk()] = word_penalty + unk_penalty

    def forward(self, input_token, timestep, *inputs):
        """
        Decoder step inputs correspond one-to-one to encoder outputs.
        """
        log_probs_per_model = []
        attn_weights_per_model = []
        state_outputs = []

        next_state_input = len(self.models)

        # underlying assumption is each model has same vocab_reduction_module
        vocab_reduction_module = self.models[0].decoder.vocab_reduction_module
        if vocab_reduction_module is not None:
            possible_translation_tokens = inputs[len(self.models)]
            next_state_input += 1
        else:
            possible_translation_tokens = None

        for i, model in enumerate(self.models):
            encoder_output = inputs[i]
            prev_hiddens = []
            prev_cells = []

            for _ in range(len(model.decoder.layers)):
                prev_hiddens.append(inputs[next_state_input])
                prev_cells.append(inputs[next_state_input + 1])
                next_state_input += 2
            prev_input_feed = inputs[next_state_input].view(1, -1)
            next_state_input += 1

            # no batching, we only care about care about "max" length
            src_length_int = encoder_output.size()[0]
            src_length = torch.LongTensor(np.array([src_length_int]))

            # notional, not actually used for decoder computation
            src_tokens = torch.LongTensor(np.array([[0] * src_length_int]))

            encoder_out = (
                encoder_output,
                prev_hiddens,
                prev_cells,
                src_length,
                src_tokens,
            )

            # store cached states, use evaluation mode
            model.decoder._is_incremental_eval = True
            model.eval()

            # placeholder
            incremental_state = {}

            # cache previous state inputs
            utils.set_incremental_state(
                model.decoder,
                incremental_state,
                'cached_state',
                (prev_hiddens, prev_cells, prev_input_feed),
            )

            decoder_output = model.decoder(
                input_token,
                encoder_out,
                incremental_state=incremental_state,
                possible_translation_tokens=possible_translation_tokens,
            )
            logits, attn_scores, _ = decoder_output

            log_probs = F.log_softmax(logits, dim=2)

            log_probs_per_model.append(log_probs)
            attn_weights_per_model.append(attn_scores)

            (
                next_hiddens,
                next_cells,
                next_input_feed,
            ) = utils.get_incremental_state(
                model.decoder,
                incremental_state,
                'cached_state',
            )

            for h, c in zip(next_hiddens, next_cells):
                state_outputs.extend([h, c])
            state_outputs.append(next_input_feed)

        average_log_probs = torch.mean(
            torch.cat(log_probs_per_model, dim=0),
            dim=0,
            keepdim=True,
        )

        average_attn_weights = torch.mean(
            torch.cat(attn_weights_per_model, dim=0),
            dim=0,
            keepdim=True,
        )

        best_scores, best_tokens = torch.topk(
            average_log_probs.view(1, -1),
            k=self.beam_size,
        )

        if possible_translation_tokens is not None:
            best_tokens = possible_translation_tokens.index_select(
                dim=0,
                index=best_tokens.view(-1),
            ).view(1, -1)

        word_rewards_for_best_tokens = self.word_rewards.index_select(
            0,
            best_tokens.view(-1),
        )
        best_scores += word_rewards_for_best_tokens

        self.input_names = ['prev_token', 'timestep']
        for i in range(len(self.models)):
            self.input_names.append('fixed_input_{}'.format(i))

        if possible_translation_tokens is not None:
            self.input_names.append('possible_translation_tokens')

        outputs = [
            best_tokens,
            best_scores,
            average_attn_weights,
        ]
        self.output_names = [
            'best_tokens_indices',
            'best_scores',
            'attention_weights_average',
        ]
        for i, state in enumerate(state_outputs):
            outputs.append(state)
            self.output_names.append('state_output_{}'.format(i))
            self.input_names.append('state_input_{}'.format(i))

        return tuple(outputs)

    def onnx_export(self, output_path, encoder_ensemble_outputs):
        # single EOS
        input_token = torch.LongTensor(
            np.array([[self.models[0].dst_dict.eos()]]),
        )
        timestep = torch.LongTensor(np.array([[0]]))

        # generate input and output names
        self.forward(input_token, timestep, *encoder_ensemble_outputs)

        onnx_export_ensemble(
            module=self,
            output_path=output_path,
            input_tuple=tuple(
                [input_token, timestep] + list(encoder_ensemble_outputs),
            ),
            input_names=self.input_names,
            output_names=self.output_names,
        )

    @staticmethod
    def build_from_checkpoints(
        checkpoint_filenames,
        src_dict_filename,
        dst_dict_filename,
        beam_size=1,
        word_penalty=0,
        unk_penalty=0,
    ):
        models = load_models_from_checkpoints(
            checkpoint_filenames,
            src_dict_filename,
            dst_dict_filename,
        )
        return DecoderStepEnsemble(
            models,
            beam_size=beam_size,
            word_penalty=word_penalty,
            unk_penalty=unk_penalty,
        )

    def save_to_db(self, output_path, encoder_ensemble_outputs):
        """
        Save encapsulated decoder step export file.
        Example encoder_ensemble_outputs (PyTorch tensors) from corresponding
        encoder are necessary to run through network once.
        """
        tmp_dir = tempfile.mkdtemp()
        tmp_file = os.path.join(tmp_dir, 'decoder_step.pb')
        self.onnx_export(tmp_file, encoder_ensemble_outputs)

        with open(tmp_file, 'r+b') as f:
            onnx_model = onnx.load(f)
        onnx_decoder_step = caffe2_backend.prepare(onnx_model)

        save_caffe2_rep_to_db(
            caffe2_backend_rep=onnx_decoder_step,
            output_path=output_path,
            input_names=self.input_names,
            output_names=self.output_names,
            num_workers=2 * len(self.models),
        )
