#!/usr/bin/env python3

import logging
import os
import tempfile

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.operators
from caffe2.python import core, dyndep, workspace
from caffe2.python.onnx import backend as caffe2_backend
from caffe2.python.predictor import predictor_exporter
from fairseq import utils
from pytorch_translate import char_source_model, dictionary, rnn, tasks  # noqa
from torch.onnx import ExportTypes


dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/aten:aten_op")

logger = logging.getLogger(__name__)


def onnx_export_ensemble(module, output_path, input_tuple, input_names, output_names):
    # include parameters as inputs of exported graph
    for name, _ in module.named_parameters():
        input_names.append(name)

    with open(output_path, "w+b") as netdef_file:
        torch.onnx._export(
            module,
            input_tuple,
            netdef_file,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            export_type=ExportTypes.ZIP_ARCHIVE,
        )


def load_models_from_checkpoints(
    checkpoint_filenames, src_dict_filename, dst_dict_filename, lexical_dict_paths=None
):
    src_dict = dictionary.Dictionary.load(src_dict_filename)
    dst_dict = dictionary.Dictionary.load(dst_dict_filename)
    models = []
    for filename in checkpoint_filenames:
        checkpoint_data = torch.load(filename, map_location="cpu")
        if lexical_dict_paths is not None:
            assert (
                checkpoint_data["args"].vocab_reduction_params is not None
            ), "lexical dictionaries can only be replaced in vocab-reduction models"
            checkpoint_data["args"].vocab_reduction_params[
                "lexical_dictionaries"
            ] = lexical_dict_paths
        task = tasks.DictionaryHolderTask(src_dict, dst_dict)
        if checkpoint_data["args"].arch == "char_source":
            model = char_source_model.CharSourceModel.build_model(
                checkpoint_data["args"], task
            )
        else:
            model = rnn.RNNModel.build_model(checkpoint_data["args"], task)
        model.load_state_dict(checkpoint_data["model"])
        models.append(model)

    return models, src_dict, dst_dict


def save_caffe2_rep_to_db(
    caffe2_backend_rep, output_path, input_names, output_names, num_workers
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

    param_names = [blob for blob in actual_external_inputs if blob not in input_names]

    init_net = core.Net(caffe2_backend_rep.init_net)
    predict_net = core.Net(caffe2_backend_rep.predict_net)

    # predictor_exporter requires disjoint params, inputs and outputs
    for i, param in enumerate(param_names):
        if param in output_names:
            saved_name = param + "_PARAM"
            init_net.Copy(param, saved_name)
            predict_net.Copy(saved_name, param)
            param_names[i] = saved_name

    output_shapes = {}
    for blob in output_names:
        output_shapes[blob] = (0,)

    # Required because of https://github.com/pytorch/pytorch/pull/6456/files
    with caffe2_backend_rep.workspace._ctx:
        workspace.RunNetOnce(init_net)
        predictor_export_meta = predictor_exporter.PredictorExportMeta(
            predict_net=predict_net,
            parameters=param_names,
            inputs=input_names,
            outputs=output_names,
            shapes=output_shapes,
            net_type="dag",
            num_workers=num_workers,
        )
        predictor_exporter.save_to_db(
            db_type="minidb",
            db_destination=output_path,
            predictor_export_meta=predictor_export_meta,
        )
    logger.info(f"Caffe2 predictor net saved as: {output_path}")


class EncoderEnsemble(nn.Module):
    def __init__(self, models, src_dict=None):
        super().__init__()
        self.models = models
        self.src_dict = src_dict
        for i, model in enumerate(self.models):
            self._modules[f"model_{i}"] = model

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
            output_names.append(f"encoder_output_{i}")

            init_hiddens, init_cells = model.decoder._init_prev_states(encoder_out)
            for h, c in zip(init_hiddens, init_cells):
                states.extend([h, c])
            states.append(model.decoder.initial_attn_context)

        # underlying assumption is each model has same vocab_reduction_module
        vocab_reduction_module = self.models[0].decoder.vocab_reduction_module
        if vocab_reduction_module is not None:
            possible_translation_tokens = vocab_reduction_module(
                src_tokens=src_tokens, decoder_input_tokens=None
            )
            outputs.append(possible_translation_tokens)
            output_names.append("possible_translation_tokens")

        for i, state in enumerate(states):
            outputs.append(state)
            output_names.append(f"initial_state_{i}")

        self.output_names = output_names

        return tuple(outputs)

    def onnx_export(self, output_path):
        # The discrepancy in types here is a temporary expedient.
        # PyTorch indexing requires int64 while support for tracing
        # pack_padded_sequence() requires int32.
        length = 5
        src_tokens = torch.LongTensor(np.ones((length, 1), dtype="int64"))
        src_lengths = torch.IntTensor(np.array([length], dtype="int32"))

        # generate output names
        self.forward(src_tokens, src_lengths)

        onnx_export_ensemble(
            module=self,
            output_path=output_path,
            input_tuple=(src_tokens, src_lengths),
            input_names=["encoder_inputs", "encoder_lengths"],
            output_names=self.output_names,
        )

    def save_to_db(self, output_path):
        """
        Save encapsulated encoder export file.
        """
        tmp_dir = tempfile.mkdtemp()
        tmp_file = os.path.join(tmp_dir, "encoder.pb")
        self.onnx_export(tmp_file)

        onnx_encoder = caffe2_backend.prepare_zip_archive(tmp_file)

        save_caffe2_rep_to_db(
            caffe2_backend_rep=onnx_encoder,
            output_path=output_path,
            input_names=["encoder_inputs", "encoder_lengths"],
            output_names=self.output_names,
            num_workers=2 * len(self.models),
        )

    @staticmethod
    def build_from_checkpoints(
        checkpoint_filenames,
        src_dict_filename,
        dst_dict_filename,
        lexical_dict_paths=None,
    ):
        models, src_dict, _ = load_models_from_checkpoints(
            checkpoint_filenames,
            src_dict_filename,
            dst_dict_filename,
            lexical_dict_paths,
        )
        return EncoderEnsemble(models, src_dict=src_dict)


class DecoderBatchedStepEnsemble(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size,
        word_reward=0,
        unk_reward=0,
        tile_internal=False,
    ):
        super().__init__()
        self.models = models
        for i, model in enumerate(self.models):
            model.decoder.attention.src_length_masking = False
            self._modules[f"model_{i}"] = model

        self.tgt_dict = tgt_dict
        self.beam_size = beam_size
        self.word_reward = word_reward
        self.unk_reward = unk_reward

        vocab_size = len(tgt_dict.indices)
        self.word_rewards = torch.FloatTensor(vocab_size).fill_(word_reward)
        self.word_rewards[tgt_dict.eos()] = 0
        self.word_rewards[tgt_dict.unk()] = word_reward + unk_reward

        self.tile_internal = tile_internal

    def forward(self, input_tokens, prev_scores, timestep, *inputs):
        """
        Decoder step inputs correspond one-to-one to encoder outputs.
        HOWEVER: after the first step, encoder outputs (i.e, the first
        len(self.models) elements of inputs) must be tiled k (beam size)
        times on the batch dimension (axis 1).
        """
        log_probs_per_model = []
        attn_weights_per_model = []
        state_outputs = []

        # from flat to (batch x 1)
        input_tokens = input_tokens.unsqueeze(1)

        next_state_input = len(self.models)

        # size of "batch" dimension of input as tensor
        batch_size = torch.onnx.operators.shape_as_tensor(input_tokens)[0]

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

            # ensure previous attention context has batch dimension
            input_feed_shape = torch.cat((batch_size.view(1), torch.LongTensor([-1])))
            prev_input_feed = torch.onnx.operators.reshape_from_tensor_shape(
                inputs[next_state_input], input_feed_shape
            )
            next_state_input += 1

            # no batching, we only care about care about "max" length
            src_length_int = int(encoder_output.size()[0])
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
                "cached_state",
                (prev_hiddens, prev_cells, prev_input_feed),
            )

            decoder_output = model.decoder(
                input_tokens,
                encoder_out,
                incremental_state=incremental_state,
                possible_translation_tokens=possible_translation_tokens,
            )
            logits, attn_scores, _ = decoder_output

            log_probs = F.log_softmax(logits, dim=2)

            log_probs_per_model.append(log_probs)
            attn_weights_per_model.append(attn_scores)

            (next_hiddens, next_cells, next_input_feed) = utils.get_incremental_state(
                model.decoder, incremental_state, "cached_state"
            )

            for h, c in zip(next_hiddens, next_cells):
                state_outputs.extend([h, c])
            state_outputs.append(next_input_feed)

        average_log_probs = torch.mean(
            torch.cat(log_probs_per_model, dim=1), dim=1, keepdim=True
        )

        average_attn_weights = torch.mean(
            torch.cat(attn_weights_per_model, dim=1), dim=1, keepdim=True
        )

        best_scores_k_by_k, best_tokens_k_by_k = torch.topk(
            average_log_probs.squeeze(1), k=self.beam_size
        )

        prev_scores_k_by_k = prev_scores.view(-1, 1).expand(-1, self.beam_size)
        total_scores_k_by_k = best_scores_k_by_k + prev_scores_k_by_k

        # flatten to take top k over all (beam x beam) hypos
        total_scores_flat = total_scores_k_by_k.view(-1)
        best_tokens_flat = best_tokens_k_by_k.view(-1)

        best_scores, best_indices = torch.topk(total_scores_flat, k=self.beam_size)

        best_tokens = best_tokens_flat.index_select(dim=0, index=best_indices).view(-1)

        # integer division to determine which input produced each successor
        prev_hypos = best_indices / self.beam_size

        attention_weights = average_attn_weights.index_select(dim=0, index=prev_hypos)

        if possible_translation_tokens is not None:
            best_tokens = possible_translation_tokens.index_select(
                dim=0, index=best_tokens
            )

        word_rewards_for_best_tokens = self.word_rewards.index_select(0, best_tokens)
        best_scores += word_rewards_for_best_tokens

        self.input_names = ["prev_tokens", "prev_scores", "timestep"]
        for i in range(len(self.models)):
            self.input_names.append(f"fixed_input_{i}")

        if possible_translation_tokens is not None:
            self.input_names.append("possible_translation_tokens")

        # 'attention_weights_average' output shape: (src_length x beam_size)
        attention_weights = attention_weights.squeeze(1)

        outputs = [best_tokens, best_scores, prev_hypos, attention_weights]
        self.output_names = [
            "best_tokens_indices",
            "best_scores",
            "prev_hypos_indices",
            "attention_weights_average",
        ]
        for i in range(len(self.models)):
            self.output_names.append(f"fixed_input_{i}")
            if self.tile_internal:
                outputs.append(inputs[i].repeat(1, self.beam_size, 1))
            else:
                outputs.append(inputs[i])

        if possible_translation_tokens is not None:
            self.output_names.append("possible_translation_tokens")
            outputs.append(possible_translation_tokens)

        for i, state in enumerate(state_outputs):
            next_state = state.index_select(dim=0, index=prev_hypos)
            outputs.append(next_state)
            self.output_names.append(f"state_output_{i}")
            self.input_names.append(f"state_input_{i}")

        return tuple(outputs)

    def onnx_export(self, output_path, encoder_ensemble_outputs):
        # single EOS (as flat array)
        input_token = torch.LongTensor(np.array([self.tgt_dict.eos()]))
        prev_scores = torch.FloatTensor(np.array([0.0]))
        timestep = torch.LongTensor(np.array([0]))

        # generate input and output names
        self.forward(input_token, prev_scores, timestep, *encoder_ensemble_outputs)

        onnx_export_ensemble(
            module=self,
            output_path=output_path,
            input_tuple=tuple(
                [input_token, prev_scores, timestep] + list(encoder_ensemble_outputs)
            ),
            input_names=self.input_names,
            output_names=self.output_names,
        )

    @staticmethod
    def build_from_checkpoints(
        checkpoint_filenames,
        src_dict_filename,
        dst_dict_filename,
        beam_size,
        word_reward=0,
        unk_reward=0,
        lexical_dict_paths=None,
    ):
        models, _, tgt_dict = load_models_from_checkpoints(
            checkpoint_filenames,
            src_dict_filename,
            dst_dict_filename,
            lexical_dict_paths,
        )
        return DecoderBatchedStepEnsemble(
            models,
            tgt_dict,
            beam_size=beam_size,
            word_reward=word_reward,
            unk_reward=unk_reward,
        )

    def save_to_db(self, output_path, encoder_ensemble_outputs):
        """
        Save encapsulated decoder step export file.
        Example encoder_ensemble_outputs (PyTorch tensors) from corresponding
        encoder are necessary to run through network once.
        """
        tmp_dir = tempfile.mkdtemp()
        tmp_file = os.path.join(tmp_dir, "decoder_step.pb")
        self.onnx_export(tmp_file, encoder_ensemble_outputs)

        onnx_decoder_step = caffe2_backend.prepare_zip_archive(tmp_file)

        save_caffe2_rep_to_db(
            caffe2_backend_rep=onnx_decoder_step,
            output_path=output_path,
            input_names=self.input_names,
            output_names=self.output_names,
            num_workers=2 * len(self.models),
        )


class BeamSearch(torch.jit.ScriptModule):

    __constants__ = ["beam_size"]

    def __init__(
        self,
        model_list,
        tgt_dict,
        src_tokens,
        src_lengths,
        beam_size=1,
        word_reward=0,
        unk_reward=0,
    ):
        super().__init__()
        self.models = model_list
        self.tgt_dict = tgt_dict
        self.beam_size = beam_size
        self.word_reward = word_reward
        self.unk_reward = unk_reward

        encoder_ens = EncoderEnsemble(self.models)
        example_encoder_outs = encoder_ens(src_tokens, src_lengths)
        self.encoder_ens = torch.jit.trace(src_tokens, src_lengths)(encoder_ens)
        decoder_ens = DecoderBatchedStepEnsemble(
            self.models,
            tgt_dict,
            beam_size,
            word_reward,
            unk_reward,
            tile_internal=False,
        )
        decoder_ens_tile = DecoderBatchedStepEnsemble(
            self.models,
            tgt_dict,
            beam_size,
            word_reward,
            unk_reward,
            tile_internal=True,
        )
        prev_token = torch.LongTensor([0])
        prev_scores = torch.FloatTensor([0.0])
        ts = torch.LongTensor([0])
        _, _, _, _, *tiled_states = decoder_ens_tile(
            prev_token, prev_scores, ts, *example_encoder_outs
        )
        self.decoder_ens_tile = torch.jit.trace(
            prev_token, prev_scores, ts, *example_encoder_outs
        )(decoder_ens_tile)
        self.decoder_ens = torch.jit.trace(
            prev_token.repeat(self.beam_size),
            prev_scores.repeat(self.beam_size),
            ts,
            *tiled_states,
        )(decoder_ens)

        self.input_names = [
            "src_tokens",
            "src_lengths",
            "prev_token",
            "prev_scores",
            "attn_weights",
            "prev_hypos_indices",
            "num_steps",
        ]
        self.output_names = [
            "all_tokens",
            "all_scores",
            "all_weights",
            "all_prev_indices",
        ]

    @torch.jit.script_method
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_token,
        prev_scores,
        attn_weights,
        prev_hypos_indices,
        num_steps,
    ):
        enc_states = self.encoder_ens(src_tokens, src_lengths)

        all_tokens = prev_token.repeat(repeats=[self.beam_size]).unsqueeze(dim=0)
        all_scores = prev_scores.repeat(repeats=[self.beam_size]).unsqueeze(dim=0)
        all_weights = (
            attn_weights.unsqueeze(dim=0)
            .repeat(repeats=[self.beam_size, 1])
            .unsqueeze(dim=0)
        )
        all_prev_indices = prev_hypos_indices.unsqueeze(dim=0)

        prev_token, prev_scores, prev_hypos_indices, attn_weights, *states = self.decoder_ens_tile(
            prev_token, prev_scores, 0, *enc_states
        )

        all_tokens = torch.cat((all_tokens, prev_token.unsqueeze(dim=0)), dim=0)
        all_scores = torch.cat((all_scores, prev_scores.unsqueeze(dim=0)), dim=0)
        all_weights = torch.cat((all_weights, attn_weights.unsqueeze(dim=0)), dim=0)
        all_prev_indices = torch.cat(
            (all_prev_indices, prev_hypos_indices.unsqueeze(dim=0)), dim=0
        )

        for i in range(num_steps - 1):
            (
                prev_token,
                prev_scores,
                prev_hypos_indices,
                attn_weights,
                *states,
            ) = self.decoder_ens(prev_token, prev_scores, i + 1, *states)

            all_tokens = torch.cat((all_tokens, prev_token.unsqueeze(dim=0)), dim=0)
            all_scores = torch.cat((all_scores, prev_scores.unsqueeze(dim=0)), dim=0)
            all_weights = torch.cat((all_weights, attn_weights.unsqueeze(dim=0)), dim=0)
            all_prev_indices = torch.cat(
                (all_prev_indices, prev_hypos_indices.unsqueeze(dim=0)), dim=0
            )

        return all_tokens, all_scores, all_weights, all_prev_indices

    def onnx_export(self, output_path):
        length = 10
        src_tokens = torch.LongTensor(np.ones((length, 1), dtype="int64"))
        src_lengths = torch.IntTensor(np.array([length], dtype="int32"))
        prev_token = torch.LongTensor([self.tgt_dict.eos()])
        prev_scores = torch.FloatTensor([0.0])
        attn_weights = torch.zeros(length)
        prev_hypos_indices = torch.zeros(self.beam_size, dtype=torch.int64)
        num_steps = torch.LongTensor([20])

        input_tuple = (
            src_tokens,
            src_lengths,
            prev_token,
            prev_scores,
            attn_weights,
            prev_hypos_indices,
            num_steps,
        )

        example_outputs = self.forward(*input_tuple)

        with open(output_path, "w+b") as netdef_file:
            torch.onnx._export(
                self,
                input_tuple,
                netdef_file,
                verbose=False,
                input_names=self.input_names,
                output_names=self.output_names,
                example_outputs=example_outputs,
                export_type=ExportTypes.ZIP_ARCHIVE,
            )

    @staticmethod
    def build_from_checkpoints(
        checkpoint_filenames,
        src_dict_filename,
        dst_dict_filename,
        beam_size,
        word_reward=0,
        unk_reward=0,
        lexical_dict_paths=None,
    ):
        length = 10
        models, _, tgt_dict = load_models_from_checkpoints(
            checkpoint_filenames,
            src_dict_filename,
            dst_dict_filename,
            lexical_dict_paths,
        )
        src_tokens = torch.LongTensor(np.ones((length, 1), dtype="int64"))
        src_lengths = torch.IntTensor(np.array([length], dtype="int32"))
        return BeamSearch(
            models,
            tgt_dict,
            src_tokens,
            src_lengths,
            beam_size=beam_size,
            word_reward=word_reward,
            unk_reward=unk_reward,
        )

    def save_to_db(self, output_path):
        """
        Save encapsulated beam search.
        """
        tmp_dir = tempfile.mkdtemp()
        tmp_file = os.path.join(tmp_dir, "beam_search.pb")
        self.onnx_export(tmp_file)

        beam_search = caffe2_backend.prepare_zip_archive(tmp_file, no_check_UNSAFE=True)

        save_caffe2_rep_to_db(
            caffe2_backend_rep=beam_search,
            output_path=output_path,
            input_names=self.input_names,
            output_names=self.output_names,
            num_workers=2 * len(self.models),
        )


class KnownOutputDecoderStepEnsemble(nn.Module):
    def __init__(self, models, tgt_dict, word_reward=0, unk_reward=0):
        super().__init__()
        self.models = models
        self.tgt_dict = tgt_dict
        for i, model in enumerate(self.models):
            if isinstance(model.encoder, char_source_model.CharRNNEncoder):
                model.encoder.onnx_export_model = True
            model.decoder.attention.src_length_masking = False
            self._modules[f"model_{i}"] = model

        self.word_reward = word_reward
        self.unk_reward = unk_reward

        vocab_size = len(tgt_dict.indices)
        self.word_rewards = torch.FloatTensor(vocab_size).fill_(word_reward)
        self.word_rewards[tgt_dict.eos()] = 0
        self.word_rewards[tgt_dict.unk()] = word_reward + unk_reward
        self.vocab_size = vocab_size
        self.unk_token = tgt_dict.unk()

    def forward(self, input_token, target_token, timestep, *inputs):
        """
        Decoder step inputs correspond one-to-one to encoder outputs.
        """
        log_probs_per_model = []
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
            src_length_int = int(encoder_output.size()[0])
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
                "cached_state",
                (prev_hiddens, prev_cells, prev_input_feed),
            )

            decoder_output = model.decoder(
                input_token.view(1, 1),
                encoder_out,
                incremental_state=incremental_state,
                possible_translation_tokens=possible_translation_tokens,
            )
            logits, _, _ = decoder_output

            log_probs = F.log_softmax(logits, dim=2)

            log_probs_per_model.append(log_probs)

            (next_hiddens, next_cells, next_input_feed) = utils.get_incremental_state(
                model.decoder, incremental_state, "cached_state"
            )

            for h, c in zip(next_hiddens, next_cells):
                state_outputs.extend([h, c])
            state_outputs.append(next_input_feed)

        average_log_probs = torch.mean(
            torch.cat(log_probs_per_model, dim=0), dim=0, keepdim=True
        )

        if possible_translation_tokens is not None:
            reduced_indices = torch.zeros(self.vocab_size).long().fill_(self.unk_token)
            # ONNX-exportable arange (ATen op)
            possible_translation_token_range = torch._dim_arange(
                like=possible_translation_tokens, dim=0
            )
            reduced_indices[
                possible_translation_tokens
            ] = possible_translation_token_range
            reduced_index = reduced_indices.index_select(dim=0, index=target_token)
            score = average_log_probs.view((-1,)).index_select(
                dim=0, index=reduced_index
            )
        else:
            score = average_log_probs.view((-1,)).index_select(
                dim=0, index=target_token
            )

        word_reward = self.word_rewards.index_select(0, target_token)
        score += word_reward

        self.input_names = ["prev_token", "target_token", "timestep"]
        for i in range(len(self.models)):
            self.input_names.append(f"fixed_input_{i}")

        if possible_translation_tokens is not None:
            self.input_names.append("possible_translation_tokens")

        outputs = [score]
        self.output_names = ["score"]

        for i in range(len(self.models)):
            self.output_names.append(f"fixed_input_{i}")
            outputs.append(inputs[i])

        if possible_translation_tokens is not None:
            self.output_names.append("possible_translation_tokens")
            outputs.append(possible_translation_tokens)

        for i, state in enumerate(state_outputs):
            outputs.append(state)
            self.output_names.append(f"state_output_{i}")
            self.input_names.append(f"state_input_{i}")

        return tuple(outputs)


class ForcedDecoder(torch.jit.ScriptModule):
    def __init__(self, model_list, tgt_dict, word_reward=0, unk_reward=0):
        super().__init__()
        self.models = model_list
        self.tgt_dict = tgt_dict
        self.word_reward = word_reward
        self.unk_reward = unk_reward

        source_tokens = torch.LongTensor(np.ones((5, 1), dtype="int64"))
        source_length = torch.LongTensor([5])

        encoder_ens = EncoderEnsemble(self.models)
        example_encoder_outs = encoder_ens(source_tokens, source_length)
        self.encoder_ens = torch.jit.trace(source_tokens, source_length)(encoder_ens)
        decoder_ens = KnownOutputDecoderStepEnsemble(
            self.models, tgt_dict, word_reward, unk_reward
        )
        prev_token = torch.LongTensor([0])
        target_token = torch.LongTensor([0])
        ts = torch.LongTensor([0])
        _, *states = decoder_ens(prev_token, target_token, ts, *example_encoder_outs)
        self.decoder_ens = torch.jit.trace(
            prev_token, target_token, ts, *example_encoder_outs
        )(decoder_ens)

        self.input_names = [
            "source_tokens",
            "source_length",
            "target_tokens",
            "target_length",
            "eos_token",
            "zero",
        ]
        self.output_names = ["score"]

    @torch.jit.script_method
    def forward(
        self,
        source_tokens,
        source_length,
        target_tokens,
        target_length,
        eos_token,
        zero,
    ):
        # EncoderEnsemble expects tokens in sequence_length-first shape
        source_tokens = source_tokens.view((-1, 1))
        states = self.encoder_ens(source_tokens, source_length)

        target_tokens = target_tokens.view((1, -1))
        eos_token = eos_token.view((1, 1))
        input_tokens = torch.cat([eos_token, target_tokens], dim=1)
        output_tokens = torch.cat([target_tokens, eos_token], dim=1)

        num_steps = int(target_length + 1)
        score = zero

        for i in range(num_steps):
            # Lint error expected (see @jamesreed's comment on D9021140)
            index_t = _to_tensor(i)
            (step_score, *states) = self.decoder_ens(
                input_tokens.index_select(dim=1, index=index_t).view((1, 1)),
                output_tokens.index_select(dim=1, index=index_t).view((1,)),
                index_t,
                *states,
            )
            score += step_score

        return score

    def onnx_export(self, output_path):
        source_tokens = torch.LongTensor(np.ones((1, 5), dtype="int64"))
        source_length = torch.LongTensor([5])
        target_tokens = torch.LongTensor(np.ones((1, 7), dtype="int64"))
        target_length = torch.LongTensor([7])
        eos_token = torch.LongTensor([[self.tgt_dict.eos()]])
        zero = torch.FloatTensor([0.0])

        input_tuple = (
            source_tokens,
            source_length,
            target_tokens,
            target_length,
            eos_token,
            zero,
        )

        example_outputs = self.forward(*input_tuple)

        with open(output_path, "w+b") as netdef_file:
            torch.onnx._export(
                self,
                input_tuple,
                netdef_file,
                verbose=False,
                input_names=self.input_names,
                output_names=self.output_names,
                example_outputs=example_outputs,
                export_type=ExportTypes.ZIP_ARCHIVE,
            )

    @staticmethod
    def build_from_checkpoints(
        checkpoint_filenames,
        src_dict_filename,
        dst_dict_filename,
        word_reward=0,
        unk_reward=0,
        lexical_dict_paths=None,
    ):
        models, _, tgt_dict = load_models_from_checkpoints(
            checkpoint_filenames,
            src_dict_filename,
            dst_dict_filename,
            lexical_dict_paths,
        )
        return ForcedDecoder(
            models, tgt_dict, word_reward=word_reward, unk_reward=unk_reward
        )

    def save_to_db(self, output_path):
        """
        Save encapsulated beam search.
        """
        tmp_dir = tempfile.mkdtemp()
        tmp_file = os.path.join(tmp_dir, "forced_decoder.pb")
        self.onnx_export(tmp_file)

        forced_decoder = caffe2_backend.prepare_zip_archive(tmp_file)

        save_caffe2_rep_to_db(
            caffe2_backend_rep=forced_decoder,
            output_path=output_path,
            input_names=self.input_names,
            output_names=self.output_names,
            num_workers=2 * len(self.models),
        )


class CharSourceEncoderEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models
        for i, model in enumerate(self.models):
            if isinstance(model.encoder, char_source_model.CharRNNEncoder):
                model.encoder.onnx_export_model = True
            self._modules[f"model_{i}"] = model

    def forward(self, src_tokens, src_lengths, char_inds, word_lengths):
        outputs = []
        output_names = []
        states = []

        # (seq_length, batch_size) for compatibility with Caffe2
        src_tokens_seq_first = src_tokens.t()

        for i, model in enumerate(self.models):
            # evaluation mode
            model.eval()

            encoder_out = model.encoder(
                src_tokens_seq_first, src_lengths, char_inds, word_lengths
            )

            # evaluation mode
            model.eval()

            # "primary" encoder output (vector representations per source token)
            encoder_outputs = encoder_out[0]
            outputs.append(encoder_outputs)
            output_names.append(f"encoder_output_{i}")

            init_hiddens, init_cells = model.decoder._init_prev_states(encoder_out)
            for h, c in zip(init_hiddens, init_cells):
                states.extend([h, c])
            states.append(model.decoder.initial_attn_context)

        # underlying assumption is each model has same vocab_reduction_module
        vocab_reduction_module = self.models[0].decoder.vocab_reduction_module
        if vocab_reduction_module is not None:
            possible_translation_tokens = vocab_reduction_module(
                src_tokens=src_tokens, decoder_input_tokens=None
            )
            outputs.append(possible_translation_tokens)
            output_names.append("possible_translation_tokens")

        for i, state in enumerate(states):
            outputs.append(state)
            output_names.append(f"initial_state_{i}")

        self.output_names = output_names

        return tuple(outputs)

    def onnx_export(self, output_path):
        # The discrepancy in types here is a temporary expedient.
        # PyTorch indexing requires int64 while support for tracing
        # pack_padded_sequence() requires int32.
        length = 5
        src_tokens = torch.LongTensor(np.ones((length, 1), dtype="int64"))
        src_lengths = torch.IntTensor(np.array([length], dtype="int32"))
        word_length = 3
        char_inds = torch.LongTensor(np.ones((1, length, word_length), dtype="int64"))
        word_lengths = torch.IntTensor(
            np.array([word_length] * length, dtype="int32")
        ).reshape((1, length))

        # generate output names
        self.forward(src_tokens, src_lengths, char_inds, word_lengths)

        onnx_export_ensemble(
            module=self,
            output_path=output_path,
            input_tuple=(src_tokens, src_lengths, char_inds, word_lengths),
            input_names=["src_tokens", "src_lengths", "char_inds", "word_lengths"],
            output_names=self.output_names,
        )

    @staticmethod
    def build_from_checkpoints(
        checkpoint_filenames,
        src_dict_filename,
        dst_dict_filename,
        lexical_dict_paths=None,
    ):
        models, _, _ = load_models_from_checkpoints(
            checkpoint_filenames,
            src_dict_filename,
            dst_dict_filename,
            lexical_dict_paths,
        )
        return CharSourceEncoderEnsemble(models)

    def save_to_db(self, output_path):
        """
        Save encapsulated encoder export file.
        """
        tmp_dir = tempfile.mkdtemp()
        tmp_file = os.path.join(tmp_dir, "encoder.pb")
        self.onnx_export(tmp_file)

        onnx_encoder = caffe2_backend.prepare_zip_archive(tmp_file)

        save_caffe2_rep_to_db(
            caffe2_backend_rep=onnx_encoder,
            output_path=output_path,
            input_names=["src_tokens", "src_lengths", "char_inds", "word_lengths"],
            output_names=self.output_names,
            num_workers=2 * len(self.models),
        )
