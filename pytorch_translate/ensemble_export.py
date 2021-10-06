#!/usr/bin/env python3

import copy
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.jit
import torch.jit.quantized
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.operators
from fairseq import tasks, utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import ARCH_MODEL_REGISTRY
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.model_utils import script_skip_tensor
from pytorch_translate.beam_decode import BeamDecode
from pytorch_translate.checkpoint import load_to_cpu, load_to_gpu
from pytorch_translate.data import dictionary
from pytorch_translate.research.knowledge_distillation import (
    dual_decoder_kd_model,
    hybrid_dual_decoder_kd_model,
)
from pytorch_translate.tasks.pytorch_translate_task import DictionaryHolderTask
from pytorch_translate.word_prediction import word_prediction_model
from torch import Tensor


try:
    from fairseq.models import fb_levenshtein_transformer as levenshtein_transformer
    from pytorch_translate import latent_var_models  # noqa;
except ImportError:
    pass


from pytorch_translate import (  # noqa; noqa
    char_aware_hybrid,
    char_source_hybrid,
    char_source_model,
    char_source_transformer_model,
    hybrid_transformer_rnn,
    rnn,
    semi_supervised,
    transformer,
)

logger = logging.getLogger(__name__)


def load_models_from_checkpoints(
    checkpoint_filenames,
    src_dict_filename,
    dst_dict_filename,
    lexical_dict_paths=None,
    use_cuda=False,
):
    src_dict = dictionary.Dictionary.load(src_dict_filename)
    dst_dict = dictionary.Dictionary.load(dst_dict_filename)
    models = []
    for filename in checkpoint_filenames:
        if use_cuda:
            checkpoint_data = load_to_gpu(filename)
        else:
            checkpoint_data = load_to_cpu(filename)
        if lexical_dict_paths is not None:
            assert (
                checkpoint_data["args"].vocab_reduction_params is not None
            ), "lexical dictionaries can only be replaced in vocab-reduction models"
            checkpoint_data["args"].vocab_reduction_params[
                "lexical_dictionaries"
            ] = lexical_dict_paths
        task = DictionaryHolderTask(src_dict, dst_dict)

        architecture = checkpoint_data["args"].arch
        if architecture == "rnn":
            model = rnn.RNNModel.build_model(checkpoint_data["args"], task)
        elif architecture == "char_source":
            model = char_source_model.CharSourceModel.build_model(
                checkpoint_data["args"], task
            )
        elif architecture == "char_source_transformer":
            model = (
                char_source_transformer_model.CharSourceTransformerModel.build_model(
                    checkpoint_data["args"], task
                )
            )
        elif architecture == "rnn_word_pred":
            model = word_prediction_model.RNNWordPredictionModel.build_model(
                checkpoint_data["args"], task
            )
        elif architecture == "ptt_transformer":
            model = transformer.TransformerModel.build_model(
                checkpoint_data["args"], task
            )
        elif architecture == "hybrid_transformer_rnn":
            model = hybrid_transformer_rnn.HybridTransformerRNNModel.build_model(
                checkpoint_data["args"], task
            )
        elif architecture == "char_source_hybrid":
            model = char_source_hybrid.CharSourceHybridModel.build_model(
                checkpoint_data["args"], task
            )
        elif architecture == "dual_decoder_kd":
            model = dual_decoder_kd_model.DualDecoderKDModel.build_model(
                checkpoint_data["args"], task
            )
        elif architecture == "hybrid_dual_decoder_kd":
            model = hybrid_dual_decoder_kd_model.HybridDualDecoderKDModel.build_model(
                checkpoint_data["args"], task
            )
        elif "semi_supervised" in architecture:
            model_args = copy.deepcopy(checkpoint_data["args"])
            model_args.source_vocab_file = src_dict_filename
            model_args.target_vocab_file = dst_dict_filename
            task = tasks.setup_task(model_args)

            model = ARCH_MODEL_REGISTRY[model_args.arch].build_model(model_args, task)
        elif architecture == "latent_var_transformer":
            task = tasks.setup_task(checkpoint_data["args"])
            model = latent_var_models.LatentVarModel.build_model(
                checkpoint_data["args"], task
            )
        elif architecture == "fb_levenshtein_transformer":
            task = tasks.setup_task(checkpoint_data["args"])
            model = levenshtein_transformer.LevenshteinTransformerModel.build_model(
                checkpoint_data["args"], task
            )
        else:
            raise RuntimeError(f"Architecture not supported: {architecture}")
        model.load_state_dict(checkpoint_data["model"])
        if hasattr(model, "get_student_model"):
            model = model.get_student_model()

        if isinstance(model, semi_supervised.SemiSupervisedModel):
            if (
                model_args.source_lang is not None
                and model_args.target_lang is not None
            ):
                direction = model_args.source_lang + "-" + model_args.target_lang
            else:
                direction = "src-tgt"
            models.append(model.models[direction])
        else:
            models.append(model)

    return models, src_dict, dst_dict


class EncoderEnsemble(nn.Module):
    def __init__(self, models, src_dict=None):
        super().__init__()
        self.models = models
        self.src_dict = src_dict
        for i, model in enumerate(self.models):
            model.prepare_for_onnx_export_()
            if hasattr(model, "get_student_model"):
                model = model.get_student_model()
                self.models[i] = model
            self._modules[f"model_{i}"] = model

        self.enable_precompute_reduced_weights = False

    def forward(self, src_tokens, src_lengths):
        # (seq_length, batch_size) for compatibility with Caffe2
        src_tokens_seq_first = src_tokens.t()

        futures = []
        for model in self.models:
            # evaluation mode
            model.eval()

            futures.append(
                torch.jit._fork(model.encoder, src_tokens_seq_first, src_lengths)
            )

        return self.get_outputs(src_tokens, futures)

    def get_outputs(self, src_tokens, encoder_futures):
        outputs = []
        output_names = []
        states = []

        possible_translation_tokens = None

        # underlying assumption is each model has same vocab_reduction_module
        if hasattr(self.models[0].decoder, "vocab_reduction_module"):
            vocab_reduction_module = self.models[0].decoder.vocab_reduction_module
            if vocab_reduction_module is not None:
                possible_translation_tokens = vocab_reduction_module(
                    src_tokens=src_tokens, decoder_input_tokens=None
                )

        # Precompute reduced decoder weight matrices.
        # Once we have possible_translation_tokens, we need to gather rows
        # out of each output_projection_{w,b} tensor for the decoders to
        # use. We do it here because these reduced matrices are used on each
        # step of the beam search, and this turns out to be a relatively
        # expensive operation.
        reduced_weights = {}
        for i, model in enumerate(self.models):
            if (
                self.enable_precompute_reduced_weights
                and hasattr(model.decoder, "_precompute_reduced_weights")
                and possible_translation_tokens is not None
            ):
                reduced_weights[i] = torch.jit._fork(
                    model.decoder._precompute_reduced_weights,
                    possible_translation_tokens,
                )

        # XXX: This loop is where we wait() for each encoder's output to be
        # ready. If you're trying to add more ops, they should probably not
        # go in this loop!
        for i, (model, future) in enumerate(zip(self.models, encoder_futures)):
            encoder_out = torch.jit._wait(future)
            # "primary" encoder output (vector representations per source token)
            encoder_outputs = encoder_out[0]
            outputs.append(encoder_outputs)
            output_names.append(f"encoder_output_{i}")
            if hasattr(model.decoder, "_init_prev_states"):
                states.extend(model.decoder._init_prev_states(encoder_out))
            if (
                self.enable_precompute_reduced_weights
                and hasattr(model.decoder, "_precompute_reduced_weights")
                and possible_translation_tokens is not None
            ):
                states.extend(torch.jit._wait(reduced_weights[i]))

        if possible_translation_tokens is not None:
            outputs.append(possible_translation_tokens)
            output_names.append("possible_translation_tokens")

        for i, state in enumerate(states):
            outputs.append(state)
            output_names.append(f"initial_state_{i}")

        self.output_names = output_names

        return tuple(outputs)

    @classmethod
    def build_from_checkpoints(
        cls,
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
        return cls(models, src_dict=src_dict)


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
            model.prepare_for_onnx_export_()
            if hasattr(model, "get_student_model"):
                model = model.get_student_model()
                self.models[i] = model
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

        self.enable_precompute_reduced_weights = False

    def forward(self, input_tokens, prev_scores, timestep, *inputs, src_tuple=None):
        """
        Decoder step inputs correspond one-to-one to encoder outputs.
        HOWEVER: after the first step, encoder outputs (i.e, the first
        len(self.models) elements of inputs) must be tiled k (beam size)
        times on the batch dimension (axis 1).
        """
        # from flat to (batch x 1)
        input_tokens = input_tokens.unsqueeze(1)

        (
            log_probs_per_model,
            attn_weights_per_model,
            state_outputs,
            beam_axis_per_state,
            possible_translation_tokens,
        ) = self._get_decoder_outputs(
            input_tokens, prev_scores, timestep, *inputs, src_tuple=src_tuple
        )

        average_log_probs = torch.mean(
            torch.cat(log_probs_per_model, dim=1), dim=1, keepdim=True
        )

        if possible_translation_tokens is None:
            word_rewards = self.word_rewards
        else:
            word_rewards = self.word_rewards.index_select(
                0, possible_translation_tokens
            )
        word_rewards = word_rewards.unsqueeze(dim=0).unsqueeze(dim=0)

        average_log_probs_with_rewards = average_log_probs + word_rewards

        average_attn_weights = torch.mean(
            torch.cat(attn_weights_per_model, dim=1), dim=1, keepdim=True
        )

        best_scores_k_by_k, best_tokens_k_by_k = torch.topk(
            average_log_probs_with_rewards.squeeze(1), k=self.beam_size
        )

        prev_scores_k_by_k = prev_scores.view(-1, 1).expand(-1, self.beam_size)
        total_scores_k_by_k = best_scores_k_by_k + prev_scores_k_by_k

        # flatten to take top k over all (beam x beam) hypos
        total_scores_flat = total_scores_k_by_k.view(-1)
        best_tokens_flat = best_tokens_k_by_k.view(-1)

        best_scores, best_indices = torch.topk(total_scores_flat, k=self.beam_size)

        best_tokens = best_tokens_flat.index_select(dim=0, index=best_indices).view(-1)

        # integer division to determine which input produced each successor
        prev_hypos = best_indices // self.beam_size

        attention_weights = average_attn_weights.index_select(dim=0, index=prev_hypos)

        if possible_translation_tokens is not None:
            best_tokens = possible_translation_tokens.index_select(
                dim=0, index=best_tokens
            )

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
            beam_axis = beam_axis_per_state[i]
            if beam_axis is None:
                next_state = state
            else:
                next_state = state.index_select(dim=beam_axis, index=prev_hypos)
            outputs.append(next_state)
            self.output_names.append(f"state_output_{i}")
            self.input_names.append(f"state_input_{i}")

        return tuple(outputs)

    def _get_decoder_outputs(
        self, input_tokens, prev_scores, timestep, *inputs, src_tuple=None
    ):
        log_probs_per_model = []
        attn_weights_per_model = []
        state_outputs = []
        beam_axis_per_state = []
        reduced_output_weights_per_model = []

        next_state_input = len(self.models)

        # size of "batch" dimension of input as tensor
        batch_size = torch.onnx.operators.shape_as_tensor(input_tokens)[0]

        possible_translation_tokens = None
        # underlying assumption is each model has same vocab_reduction_module
        if hasattr(self.models[0].decoder, "vocab_reduction_module"):
            vocab_reduction_module = self.models[0].decoder.vocab_reduction_module
            if vocab_reduction_module is not None:
                possible_translation_tokens = inputs[len(self.models)]
                next_state_input += 1

        futures = []

        for i, model in enumerate(self.models):
            if (
                isinstance(model, rnn.RNNModel)
                or isinstance(model, rnn.DummyPyTextRNNPointerModel)
                or isinstance(model, char_source_model.CharSourceModel)
                or isinstance(model, word_prediction_model.WordPredictionModel)
            ):
                encoder_output = inputs[i]
                prev_hiddens = []
                prev_cells = []

                for _ in range(len(model.decoder.layers)):
                    prev_hiddens.append(inputs[next_state_input])
                    prev_cells.append(inputs[next_state_input + 1])
                    next_state_input += 2

                # ensure previous attention context has batch dimension
                input_feed_shape = torch.cat(
                    (batch_size.view(1), torch.LongTensor([-1]))
                )
                prev_input_feed = torch.onnx.operators.reshape_from_tensor_shape(
                    inputs[next_state_input], input_feed_shape
                )
                next_state_input += 1

                if (
                    self.enable_precompute_reduced_weights
                    and hasattr(model.decoder, "_precompute_reduced_weights")
                    and possible_translation_tokens is not None
                ):
                    # (output_projection_w, output_projection_b)
                    reduced_output_weights = inputs[
                        next_state_input : next_state_input + 2
                    ]
                    next_state_input += 2
                else:
                    reduced_output_weights = None
                reduced_output_weights_per_model.append(reduced_output_weights)

                # no batching, we only care about care about "max" length
                if src_tuple:
                    src_tokens, src_length = src_tuple
                    src_tokens = src_tokens.t()
                else:
                    # notional, not actually used for decoder computation
                    src_length_int = int(encoder_output.size()[0])
                    src_length = torch.LongTensor(np.array([src_length_int]))
                    src_tokens = torch.LongTensor(np.array([[0] * src_length_int]))
                src_embeddings = encoder_output.new_zeros(encoder_output.shape)

                encoder_out = (
                    encoder_output,
                    prev_hiddens,
                    prev_cells,
                    src_length,
                    src_tokens,
                    src_embeddings,
                )

                def forked_section(
                    input_tokens,
                    encoder_out,
                    possible_translation_tokens,
                    prev_hiddens,
                    prev_cells,
                    prev_input_feed,
                    reduced_output_weights,
                ):
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
                        reduced_output_weights=reduced_output_weights,
                    )
                    logits, attn_scores, _ = decoder_output

                    log_probs = (
                        logits
                        if isinstance(model, rnn.DummyPyTextRNNPointerModel)
                        else F.log_softmax(logits, dim=2)
                    )

                    log_probs_per_model.append(log_probs)
                    attn_weights_per_model.append(attn_scores)

                    (
                        next_hiddens,
                        next_cells,
                        next_input_feed,
                    ) = utils.get_incremental_state(
                        model.decoder, incremental_state, "cached_state"
                    )

                    return (
                        log_probs,
                        attn_scores,
                        tuple(next_hiddens),
                        tuple(next_cells),
                        next_input_feed,
                    )

                fut = torch.jit._fork(
                    forked_section,
                    input_tokens,
                    encoder_out,
                    possible_translation_tokens,
                    prev_hiddens,
                    prev_cells,
                    prev_input_feed,
                    reduced_output_weights,
                )

                futures.append(fut)
            elif isinstance(model, transformer.TransformerModel) or isinstance(
                model, char_source_transformer_model.CharSourceTransformerModel
            ):
                encoder_output = inputs[i]
                # store cached states, use evaluation mode
                model.decoder._is_incremental_eval = True
                model.eval()

                states_per_layer = 4
                state_inputs = []
                for i, _ in enumerate(model.decoder.layers):
                    # (prev_key, prev_value) for self- and encoder-attention
                    if hasattr(model.decoder, "decoder_layers_to_keep") and (
                        i not in model.decoder.decoder_layers_to_keep.keys()
                    ):
                        continue

                    state_inputs.extend(
                        inputs[next_state_input : next_state_input + states_per_layer]
                    )
                    next_state_input += states_per_layer

                encoder_out = (encoder_output, None, None)

                # TODO(jcross)
                reduced_output_weights = None
                reduced_output_weights_per_model.append(reduced_output_weights)

                def forked_section(
                    input_tokens,
                    encoder_out,
                    state_inputs,
                    possible_translation_tokens,
                    timestep,
                ):
                    decoder_output = model.decoder(
                        input_tokens,
                        encoder_out,
                        incremental_state=state_inputs,
                        possible_translation_tokens=possible_translation_tokens,
                        timestep=timestep,
                    )
                    logits, attn_scores, _, attention_states = decoder_output

                    log_probs = F.log_softmax(logits, dim=2)

                    return log_probs, attn_scores, tuple(attention_states)

                fut = torch.jit._fork(
                    forked_section,
                    input_tokens,
                    encoder_out,
                    state_inputs,
                    possible_translation_tokens,
                    timestep,
                )

                futures.append(fut)

            elif isinstance(model, levenshtein_transformer.LevenshteinTransformerModel):
                encoder_output = inputs[i]
                # store cached states, use evaluation mode
                model.decoder._is_incremental_eval = True
                model.eval()

                states_per_layer = 4

                state_inputs = []
                for _ in model.decoder.layers:
                    # (prev_key, prev_value) for self- and encoder-attention
                    state_inputs.extend(
                        inputs[next_state_input : next_state_input + states_per_layer]
                    )
                    next_state_input += states_per_layer

                encoder_out = (encoder_output, None, None)

                # TODO(jcross)
                reduced_output_weights = None
                reduced_output_weights_per_model.append(reduced_output_weights)

                def forked_section(
                    input_tokens,
                    encoder_out,
                    state_inputs,
                    possible_translation_tokens,
                    timestep,
                ):
                    decoder_output = model.decoder(
                        input_tokens,
                        encoder_out,
                        incremental_state=state_inputs,
                        possible_translation_tokens=possible_translation_tokens,
                        timestep=timestep,
                    )
                    logits, attn_scores, attention_states = decoder_output

                    log_probs = F.log_softmax(logits, dim=2)

                    return log_probs, attn_scores, tuple(attention_states)

                fut = torch.jit._fork(
                    forked_section,
                    input_tokens,
                    encoder_out,
                    state_inputs,
                    possible_translation_tokens,
                    timestep,
                )

                futures.append(fut)
            elif isinstance(model, latent_var_models.LatentVarModel):
                encoder_output = inputs[i]
                # store cached states, use evaluation mode
                model.decoder._is_incremental_eval = True
                model.eval()
                state_inputs = []
                state_inputs.extend(inputs[next_state_input : next_state_input + 3])
                next_state_input += 3
                for _ in list(model.decoder.decoders.values())[0].layers:
                    # (prev_key, prev_value) for self- and encoder-attention
                    state_inputs.extend(inputs[next_state_input : next_state_input + 4])
                    next_state_input += 4

                encoder_out = encoder_output

                # TODO(jcross)
                reduced_output_weights = None
                reduced_output_weights_per_model.append(reduced_output_weights)

                def forked_section(
                    input_tokens,
                    encoder_out,
                    state_inputs,
                    possible_translation_tokens,
                    timestep,
                ):
                    decoder_output = model.decoder(
                        input_tokens, encoder_out, incremental_state=state_inputs
                    )
                    logits, attn_scores, _, _, attention_states = decoder_output

                    log_probs = F.log_softmax(logits, dim=2)

                    return log_probs, attn_scores, tuple(attention_states)

                fut = torch.jit._fork(
                    forked_section,
                    input_tokens,
                    encoder_out,
                    state_inputs,
                    possible_translation_tokens,
                    timestep,
                )

                futures.append(fut)

            elif isinstance(
                model, hybrid_transformer_rnn.HybridTransformerRNNModel
            ) or isinstance(model, char_source_hybrid.CharSourceHybridModel):
                encoder_output = inputs[i]

                # store cached states, use evaluation mode
                model.decoder._is_incremental_eval = True
                model.eval()

                encoder_out = (encoder_output, None, None)

                num_states = (1 + model.decoder.num_layers) * 2
                state_inputs = inputs[next_state_input : next_state_input + num_states]
                next_state_input += num_states

                # TODO(jcross)
                reduced_output_weights = None
                reduced_output_weights_per_model.append(reduced_output_weights)

                def forked_section(
                    input_tokens,
                    encoder_out,
                    state_inputs,
                    possible_translation_tokens,
                    timestep,
                ):
                    incremental_state = {}
                    utils.set_incremental_state(
                        model.decoder, incremental_state, "cached_state", state_inputs
                    )

                    decoder_output = model.decoder(
                        input_tokens,
                        encoder_out,
                        incremental_state=incremental_state,
                        possible_translation_tokens=possible_translation_tokens,
                        timestep=timestep,
                    )
                    logits, attn_scores, _ = decoder_output

                    log_probs = F.log_softmax(logits, dim=2)

                    next_states = utils.get_incremental_state(
                        model.decoder, incremental_state, "cached_state"
                    )

                    return log_probs, attn_scores, tuple(next_states)

                fut = torch.jit._fork(
                    forked_section,
                    input_tokens,
                    encoder_out,
                    state_inputs,
                    possible_translation_tokens,
                    timestep,
                )

                futures.append(fut)
            else:
                raise RuntimeError(f"Not a supported model: {type(model)}")

        for i, (model, fut) in enumerate(zip(self.models, futures)):
            if (
                isinstance(model, rnn.RNNModel)
                or isinstance(model, rnn.DummyPyTextRNNPointerModel)
                or isinstance(model, char_source_model.CharSourceModel)
                or isinstance(model, word_prediction_model.WordPredictionModel)
            ):
                (
                    log_probs,
                    attn_scores,
                    next_hiddens,
                    next_cells,
                    next_input_feed,
                ) = torch.jit._wait(fut)

                for h, c in zip(next_hiddens, next_cells):
                    state_outputs.extend([h, c])
                    beam_axis_per_state.extend([0, 0])

                state_outputs.append(next_input_feed)
                beam_axis_per_state.append(0)

                if reduced_output_weights_per_model[i] is not None:
                    state_outputs.extend(reduced_output_weights_per_model[i])
                    beam_axis_per_state.extend(
                        [None for _ in reduced_output_weights_per_model[i]]
                    )

            elif isinstance(model, transformer.TransformerModel) or isinstance(
                model, char_source_transformer_model.CharSourceTransformerModel
            ):
                log_probs, attn_scores, attention_states = torch.jit._wait(fut)

                log_probs_per_model.append(log_probs)
                attn_weights_per_model.append(attn_scores)

                state_outputs.extend(attention_states)
                beam_axis_per_state.extend([0 for _ in attention_states])
            elif isinstance(model, levenshtein_transformer.LevenshteinTransformerModel):
                log_probs, attn_scores, attention_states = torch.jit._wait(fut)

                log_probs_per_model.append(log_probs)
                attn_weights_per_model.append(attn_scores)

                state_outputs.extend(attention_states)
                beam_axis_per_state.extend([None for _ in attention_states])
            elif isinstance(model, latent_var_models.LatentVarModel):
                log_probs, attn_scores, attention_states = torch.jit._wait(fut)

                log_probs_per_model.append(log_probs)
                attn_weights_per_model.append(attn_scores)
                state_outputs.extend(attention_states)
                beam_axis_per_state.extend([0 for _ in attention_states])
            elif isinstance(
                model, hybrid_transformer_rnn.HybridTransformerRNNModel
            ) or isinstance(model, char_source_hybrid.CharSourceHybridModel):
                log_probs, attn_scores, next_states = torch.jit._wait(fut)

                log_probs_per_model.append(log_probs)
                attn_weights_per_model.append(attn_scores)

                state_outputs.extend(next_states)
                # sequence RNN states have beam along axis 1
                beam_axis_per_state.extend([1 for _ in next_states[:-2]])
                # encoder input projections have beam along axis 0
                beam_axis_per_state.extend([0, 0])
            else:
                raise RuntimeError(f"Not a supported model: {type(model)}")

        return (
            log_probs_per_model,
            attn_weights_per_model,
            state_outputs,
            beam_axis_per_state,
            possible_translation_tokens,
        )

    @classmethod
    def build_from_checkpoints(
        cls,
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
        return cls(
            models,
            tgt_dict,
            beam_size=beam_size,
            word_reward=word_reward,
            unk_reward=unk_reward,
        )


class FakeEncoderEnsemble(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, src_tokens, src_lengths) -> None:
        raise RuntimeError(
            "Called EncoderEnsemble on a BeamSearch thats not word-source"
        )


class FakeCharSourceEncoderEnsemble(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, src_tokens, src_lengths, char_inds, word_lengths) -> None:
        raise RuntimeError(
            "Called CharSourceEncoderEnsemble on a BeamSearch thats not char-source"
        )


class BeamSearch(torch.jit.ScriptModule):

    __constants__ = ["beam_size", "is_char_source"]

    def __init__(
        self,
        model_list,
        tgt_dict,
        src_tokens,
        src_lengths,
        beam_size=1,
        word_reward=0,
        unk_reward=0,
        quantize=False,
        # Tracing inputs for CharSourceModel
        char_inds=None,
        word_lengths=None,
    ):
        super().__init__()
        self.models = model_list
        self.tgt_dict = tgt_dict
        self.beam_size = beam_size
        self.word_reward = word_reward
        self.unk_reward = unk_reward

        if (
            isinstance(self.models[0], char_source_model.CharSourceModel)
            or isinstance(
                self.models[0], char_source_transformer_model.CharSourceTransformerModel
            )
            or isinstance(self.models[0], char_source_hybrid.CharSourceHybridModel)
        ):
            encoder_ens = CharSourceEncoderEnsemble(self.models)
        else:
            encoder_ens = EncoderEnsemble(self.models)
        encoder_ens.enable_precompute_reduced_weights = True

        if quantize:
            torch.ao.quantization.quantize_dynamic(
                encoder_ens, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
            encoder_ens = torch.jit.quantized.quantize_rnn_cell_modules(encoder_ens)

        if (
            isinstance(self.models[0], char_source_model.CharSourceModel)
            or isinstance(
                self.models[0], char_source_transformer_model.CharSourceTransformerModel
            )
            or isinstance(self.models[0], char_source_hybrid.CharSourceHybridModel)
        ):
            self.is_char_source = True
            enc_inputs = (src_tokens, src_lengths, char_inds, word_lengths)
            example_encoder_outs = encoder_ens(*enc_inputs)
            self.encoder_ens = FakeEncoderEnsemble()
            self.encoder_ens_char_source = torch.jit.trace(
                encoder_ens, enc_inputs, _force_outplace=True, check_trace=False
            )
        else:
            self.is_char_source = False
            enc_inputs = (src_tokens, src_lengths)
            example_encoder_outs = encoder_ens(*enc_inputs)
            self.encoder_ens = torch.jit.trace(
                encoder_ens, enc_inputs, _force_outplace=True, check_trace=False
            )
            self.encoder_ens_char_source = FakeCharSourceEncoderEnsemble()

        decoder_ens = DecoderBatchedStepEnsemble(
            self.models,
            tgt_dict,
            beam_size,
            word_reward,
            unk_reward,
            tile_internal=False,
        )
        decoder_ens.enable_precompute_reduced_weights = True
        if quantize:
            torch.ao.quantization.quantize_dynamic(
                decoder_ens, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
            decoder_ens = torch.jit.quantized.quantize_rnn_cell_modules(decoder_ens)
            decoder_ens = torch.jit.quantized.quantize_rnn_modules(decoder_ens)
        decoder_ens_tile = DecoderBatchedStepEnsemble(
            self.models,
            tgt_dict,
            beam_size,
            word_reward,
            unk_reward,
            tile_internal=True,
        )
        decoder_ens_tile.enable_precompute_reduced_weights = True
        if quantize:
            torch.ao.quantization.quantize_dynamic(
                decoder_ens_tile, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
            decoder_ens_tile = torch.jit.quantized.quantize_rnn_cell_modules(
                decoder_ens_tile
            )
            decoder_ens_tile = torch.jit.quantized.quantize_rnn_modules(
                decoder_ens_tile
            )
        prev_token = torch.LongTensor([0])
        prev_scores = torch.FloatTensor([0.0])
        ts = torch.LongTensor([0])
        _, _, _, _, *tiled_states = decoder_ens_tile(
            prev_token, prev_scores, ts, *example_encoder_outs
        )
        self.decoder_ens_tile = torch.jit.trace(
            decoder_ens_tile,
            (prev_token, prev_scores, ts, *example_encoder_outs),
            _force_outplace=True,
            check_trace=False,
        )
        self.decoder_ens = torch.jit.trace(
            decoder_ens,
            (
                prev_token.repeat(self.beam_size),
                prev_scores.repeat(self.beam_size),
                ts,
                *tiled_states,
            ),
            _force_outplace=True,
            check_trace=False,
        )

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
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_token: torch.Tensor,
        prev_scores: torch.Tensor,
        attn_weights: torch.Tensor,
        prev_hypos_indices: torch.Tensor,
        num_steps: int,
        char_inds: Optional[torch.Tensor] = None,
        word_lengths: Optional[torch.Tensor] = None,
    ):
        if self.is_char_source:
            if char_inds is None or word_lengths is None:
                raise RuntimeError(
                    "char_inds and word_lengths must be specified "
                    "for char-source models"
                )
            char_inds = torch.jit._unwrap_optional(char_inds)
            word_lengths = torch.jit._unwrap_optional(word_lengths)
            enc_states = self.encoder_ens_char_source(
                src_tokens, src_lengths, char_inds, word_lengths
            )
        else:
            enc_states = self.encoder_ens(src_tokens, src_lengths)

        # enc_states ends up being optional because of the above branch, one
        # side returns None. We should never take the path that returns None
        # so we unrap the optional type here.
        enc_states = torch.jit._unwrap_optional(enc_states)

        all_tokens = prev_token.repeat(repeats=[self.beam_size]).unsqueeze(dim=0)
        all_scores = prev_scores.repeat(repeats=[self.beam_size]).unsqueeze(dim=0)
        all_weights = (
            attn_weights.unsqueeze(dim=0)
            .repeat(repeats=[self.beam_size, 1])
            .unsqueeze(dim=0)
        )
        all_prev_indices = prev_hypos_indices.unsqueeze(dim=0)

        (
            prev_token,
            prev_scores,
            prev_hypos_indices,
            attn_weights,
            *states,
        ) = self.decoder_ens_tile(
            prev_token, prev_scores, _to_tensor(0), *enc_states  # noqa
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
            ) = self.decoder_ens(
                prev_token, prev_scores, _to_tensor(i + 1), *states  # noqa
            )

            all_tokens = torch.cat((all_tokens, prev_token.unsqueeze(dim=0)), dim=0)
            all_scores = torch.cat((all_scores, prev_scores.unsqueeze(dim=0)), dim=0)
            all_weights = torch.cat((all_weights, attn_weights.unsqueeze(dim=0)), dim=0)
            all_prev_indices = torch.cat(
                (all_prev_indices, prev_hypos_indices.unsqueeze(dim=0)), dim=0
            )

        return all_tokens, all_scores, all_weights, all_prev_indices

    @classmethod
    def build_from_checkpoints(
        cls,
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
        if (
            isinstance(models[0], char_source_model.CharSourceModel)
            or isinstance(
                models[0], char_source_transformer_model.CharSourceTransformerModel
            )
            or isinstance(models[0], char_source_hybrid.CharSourceHybridModel)
        ):
            word_length = 3
            char_inds = torch.LongTensor(
                np.ones((1, length, word_length), dtype="int64")
            )
            word_lengths = torch.IntTensor(
                np.array([word_length] * length, dtype="int32")
            ).reshape((1, length))
        else:
            char_inds = None
            word_lengths = None
        return cls(
            models,
            tgt_dict,
            src_tokens,
            src_lengths,
            beam_size=beam_size,
            word_reward=word_reward,
            unk_reward=unk_reward,
            quantize=True,
            char_inds=char_inds,
            word_lengths=word_lengths,
        )

    def save_to_pytorch(self, output_path):
        def pack(s):
            if hasattr(s, "_pack"):
                s._pack()

        def unpack(s):
            if hasattr(s, "_unpack"):
                s._unpack()

        self.apply(pack)
        torch.jit.save(self, output_path)
        self.apply(unpack)


class KnownOutputDecoderStepEnsemble(nn.Module):
    def __init__(self, models, tgt_dict, word_reward=0, unk_reward=0):
        super().__init__()
        self.models = models
        self.tgt_dict = tgt_dict
        for i, model in enumerate(self.models):
            model.prepare_for_onnx_export_()
            self._modules[f"model_{i}"] = model

        self.word_reward = word_reward
        self.unk_reward = unk_reward

        vocab_size = len(tgt_dict.indices)
        self.word_rewards = torch.FloatTensor(vocab_size).fill_(word_reward)
        self.word_rewards[tgt_dict.eos()] = 0
        self.word_rewards[tgt_dict.unk()] = word_reward + unk_reward
        self.vocab_size = vocab_size
        self.unk_token = tgt_dict.unk()

        self.enable_precompute_reduced_weights = False

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

            if (
                self.enable_precompute_reduced_weights
                and hasattr(model.decoder, "_precompute_reduced_weights")
                and possible_translation_tokens is not None
            ):
                # (output_projection_w, output_projection_b)
                reduced_output_weights = inputs[next_state_input : next_state_input + 2]
                next_state_input += 2
            else:
                reduced_output_weights = None

            # no batching, we only care about care about "max" length
            src_length_int = int(encoder_output.size()[0])
            src_length = torch.LongTensor(np.array([src_length_int]))

            # notional, not actually used for decoder computation
            src_tokens = torch.LongTensor(np.array([[0] * src_length_int]))
            src_embeddings = encoder_output.new_zeros(encoder_output.shape)

            encoder_out = (
                encoder_output,
                prev_hiddens,
                prev_cells,
                src_length,
                src_tokens,
                src_embeddings,
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

            if reduced_output_weights is not None:
                state_outputs.extend(reduced_output_weights)

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


class CharSourceEncoderEnsemble(nn.Module):
    def __init__(self, models, src_dict=None):
        super().__init__()
        self.models = models
        self.src_dict = src_dict
        for i, model in enumerate(self.models):
            model.prepare_for_onnx_export_()
            self._modules[f"model_{i}"] = model

        self.enable_precompute_reduced_weights = False

    def forward(self, src_tokens, src_lengths, char_inds, word_lengths):
        outputs = []
        output_names = []
        states = []

        # (seq_length, batch_size) for compatibility with Caffe2
        src_tokens_seq_first = src_tokens.t()

        futures = []
        for model in self.models:
            # evaluation mode
            model.eval()
            futures.append(
                torch.jit._fork(
                    model.encoder,
                    src_tokens_seq_first,
                    src_lengths,
                    char_inds,
                    word_lengths,
                )
            )

        # underlying assumption is each model has same vocab_reduction_module
        vocab_reduction_module = self.models[0].decoder.vocab_reduction_module
        possible_translation_tokens = None
        if vocab_reduction_module is not None:
            possible_translation_tokens = vocab_reduction_module(
                src_tokens=src_tokens, decoder_input_tokens=None
            )

        # Precompute reduced decoder weight matrices.
        # Once we have possible_translation_tokens, we need to gather rows
        # out of each output_projection_{w,b} tensor for the decoders to
        # use. We do it here because these reduced matrices are used on each
        # step of the beam search, and this turns out to be a relatively
        # expensive operation.
        reduced_weights = {}
        for i, model in enumerate(self.models):
            if (
                self.enable_precompute_reduced_weights
                and hasattr(model.decoder, "_precompute_reduced_weights")
                and possible_translation_tokens is not None
            ):
                reduced_weights[i] = torch.jit._fork(
                    model.decoder._precompute_reduced_weights,
                    possible_translation_tokens,
                )

        # XXX: This loop is where we wait() for each encoder's output to be
        # ready. If you're trying to add more ops, they should probably not
        # go in this loop!
        for i, (model, future) in enumerate(zip(self.models, futures)):
            encoder_out = torch.jit._wait(future)

            # "primary" encoder output (vector representations per source token)
            encoder_outputs = encoder_out[0]
            outputs.append(encoder_outputs)
            output_names.append(f"encoder_output_{i}")

            if hasattr(model.decoder, "_init_prev_states"):
                states.extend(model.decoder._init_prev_states(encoder_out))
            if (
                self.enable_precompute_reduced_weights
                and hasattr(model.decoder, "_precompute_reduced_weights")
                and possible_translation_tokens is not None
            ):
                states.extend(torch.jit._wait(reduced_weights[i]))

        if possible_translation_tokens is not None:
            outputs.append(possible_translation_tokens)
            output_names.append("possible_translation_tokens")

        for i, state in enumerate(states):
            outputs.append(state)
            output_names.append(f"initial_state_{i}")

        self.output_names = output_names

        return tuple(outputs)

    @classmethod
    def build_from_checkpoints(
        cls,
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
        return cls(models, src_dict=src_dict)


class BeamSearchAndDecode(torch.jit.ScriptModule):
    """
    Combines the functionality of BeamSearch and BeamDecode
    """

    def __init__(
        self,
        models,
        tgt_dict,
        src_tokens,
        src_lengths,
        eos_token_id,
        length_penalty,
        nbest,
        beam_size,
        stop_at_eos,
        word_reward=0,
        unk_reward=0,
        quantize=False,
    ):
        super().__init__()

        self.beam_search = BeamSearch(
            models,
            tgt_dict,
            src_tokens,
            src_lengths,
            beam_size,
            word_reward,
            unk_reward,
            quantize,
        )

        self.beam_decode = BeamDecode(
            eos_token_id, length_penalty, nbest, beam_size, stop_at_eos
        )

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
            "beam_output",
            "hypothesis_score",
            "token_level_scores",
            "back_alignment_weights",
            "best_indices",
        ]

    @torch.jit.script_method
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_token: torch.Tensor,
        prev_scores: torch.Tensor,
        attn_weights: torch.Tensor,
        prev_hypos_indices: torch.Tensor,
        num_steps: int,
    ) -> List[Tuple[Tensor, float, List[float], Tensor, Tensor]]:

        beam_search_out = self.beam_search(
            src_tokens,
            src_lengths,
            prev_token,
            prev_scores,
            attn_weights,
            prev_hypos_indices,
            num_steps,
        )
        all_tokens, all_scores, all_weights, all_prev_indices = beam_search_out

        outputs = torch.jit.annotate(
            List[Tuple[Tensor, float, List[float], Tensor, Tensor]], []
        )
        outputs = self.beam_decode(
            all_tokens, all_scores, all_weights, all_prev_indices, num_steps
        )

        return outputs

    @classmethod
    def build_from_checkpoints(
        cls,
        checkpoint_filenames,
        src_dict_filename,
        dst_dict_filename,
        beam_size,
        length_penalty,
        nbest,
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
        eos_token_id = tgt_dict.eos()

        return cls(
            models,
            tgt_dict,
            src_tokens,
            src_lengths,
            eos_token_id,
            length_penalty=length_penalty,
            nbest=nbest,
            beam_size=beam_size,
            stop_at_eos=True,
            word_reward=word_reward,
            unk_reward=unk_reward,
            quantize=True,
        )

    def save_to_pytorch(self, output_path):
        def pack(s):
            if hasattr(s, "_pack"):
                s._pack()

        def unpack(s):
            if hasattr(s, "_unpack"):
                s._unpack()

        self.apply(pack)
        torch.jit.save(self, output_path)
        self.apply(unpack)


@torch.jit.script
def finalize_hypos_loop_tokens(
    finalized_tokens_list: List[Tensor],
    finalized_idxs,
    pad_idx: int,
    finalized_tokens,
    finalized_scores,
):
    for i in range(finalized_idxs.size(0)):
        cutoff = finalized_tokens[i].ne(pad_idx)
        tokens = finalized_tokens[i][cutoff]
        finalized_tokens_list[finalized_idxs[i]] = tokens
    return finalized_tokens_list


@torch.jit.script
def finalize_hypos_loop_scores(
    finalized_scores_list: List[Tensor],
    finalized_idxs,
    pad_idx: int,
    finalized_tokens,
    finalized_scores,
):
    for i in range(finalized_idxs.size(0)):
        cutoff = finalized_scores[i].ne(pad_idx)
        scores = finalized_scores[i][cutoff]
        finalized_scores_list[finalized_idxs[i]] = scores
    return finalized_scores_list


@torch.jit.script
def finalize_hypos_loop_attns(
    finalized_attns_list: List[Tensor],
    finalized_alignments_list: List[Tensor],
    finalized_idxs,
    pad_idx: int,
    finalized_tokens,
    finalized_scores,
    finalized_attn,
):
    for i in range(finalized_idxs.size(0)):
        cutoff = finalized_tokens[i].ne(pad_idx)
        hypo_attn = finalized_attn[i][cutoff]
        alignment = hypo_attn.max(dim=1)[1]
        finalized_attns_list[finalized_idxs[i]] = hypo_attn
        finalized_alignments_list[finalized_idxs[i]] = alignment

    return finalized_attns_list, finalized_alignments_list


class IterativeRefinementGenerateAndDecode(torch.jit.ScriptModule):
    def __init__(self, models, tgt_dict, max_iter=1, quantize=True, check_trace=True):
        super().__init__()
        src_tokens = torch.tensor([[4, 2]])
        src_lengths = torch.tensor([2])
        self.models = models

        generator = IterativeRefinementGenerator(
            self.models, tgt_dict, max_iter=max_iter
        )
        if quantize:
            generator = torch.ao.quantization.quantize_dynamic(
                generator, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
        enc_inputs = (src_tokens, src_lengths)
        self.generator = torch.jit.trace(
            generator, enc_inputs, _force_outplace=True, check_trace=check_trace
        )

    @torch.jit.script_method
    def forward(
        self, src_tokens: torch.Tensor, src_lengths: torch.Tensor
    ) -> List[Tuple[Tensor, float, Tensor]]:

        return [
            (x.long(), float(y), at)
            for x, y, at in list(self.generator(src_tokens.t(), src_lengths))
        ]

    def save_to_pytorch(self, output_path):
        def pack(s):
            if hasattr(s, "_pack"):
                s._pack()

        def unpack(s):
            if hasattr(s, "_unpack"):
                s._unpack()

        self.apply(pack)
        torch.jit.save(self, output_path)
        self.apply(unpack)

    @classmethod
    def build_from_checkpoints(
        cls,
        checkpoint_filenames,
        src_dict_filename,
        tgt_dict_filename,
        lexical_dict_paths=None,
        max_iter=1,
    ):
        models, _, tgt_dict = load_models_from_checkpoints(
            checkpoint_filenames,
            src_dict_filename,
            tgt_dict_filename,
            lexical_dict_paths,
        )
        return cls(models, tgt_dict=tgt_dict, max_iter=max_iter)


@torch.jit.script
def is_a_loop(pad_idx: int, x, y, s, a):
    b, l_x, l_y = x.size(0), x.size(1), y.size(1)
    if l_x > l_y:
        y = torch.cat([y, torch.zeros([b, l_x - l_y]).to(y).fill_(pad_idx)], 1)
        s = torch.cat([s, torch.zeros([b, l_x - l_y]).to(s)], 1)
        if a.size()[0] > 0:
            a = torch.cat([a, torch.zeros([b, l_x - l_y, a.size(2)]).to(a)], 1)
    elif l_x < l_y:
        x = torch.cat([x, torch.zeros([b, l_y - l_x]).to(x).fill_(pad_idx)], 1)
    return (x == y).all(1), y, s, a


@torch.jit.script
def last_step(step: int, max_iter: int, terminated):
    if step == max_iter:  # reach last iteration, terminate
        terminated.fill_(1)
    return terminated


class IterativeRefinementGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        eos_penalty=0.0,
        max_iter=2,
        max_ratio=2,
        decoding_format=None,
        retain_dropout=False,
        adaptive=True,
    ):
        """
        Generates translations based on iterative refinement.

        Args:
            tgt_dict: target dictionary
        """
        super().__init__()
        self.models = models
        self.bos = tgt_dict.bos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.eos_penalty = eos_penalty
        self.max_iter = max_iter
        self.max_ratio = max_ratio
        self.decoding_format = decoding_format
        self.retain_dropout = retain_dropout
        self.adaptive = adaptive
        for i, model in enumerate(self.models):
            model.prepare_for_onnx_export_()
            model.eval()
            if hasattr(model, "get_student_model"):
                model = model.get_student_model()
                self.models[i] = model
            self._modules[f"model_{i}"] = model

    def forward(
        self, src_tokens: torch.Tensor, src_lengths: torch.Tensor
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor]]:

        o1, o2, o3, _ = self.generate(self.models, src_tokens, src_lengths)
        return tuple((x, y.float().mean(), z) for x, y, z in zip(o1, o2, o3))

    @torch.no_grad()
    def generate(self, models, src_tokens, src_lengths, prefix_tokens=None):

        # TODO: model ensemble
        assert len(models) == 1, "only support single model"
        model = models[0]
        bsz, src_len = src_tokens.size()
        sent_idxs = torch.arange(bsz)
        # encoding
        encoder_out = model.encoder(src_tokens, src_lengths)

        # initialize buffers (very model specific, with length prediction or not)
        prev_decoder_out = model.initialize_output_tokens(encoder_out, src_tokens)
        prev_output_tokens = prev_decoder_out.output_tokens.clone()

        finalized_tokens_list = [torch.tensor(0) for _ in range(bsz)]
        finalized_scores_list = [torch.tensor(0) for _ in range(bsz)]
        finalized_attns_list = [torch.tensor(0) for _ in range(bsz)]
        finalized_alignments_list = [torch.tensor(0) for _ in range(bsz)]

        for step in range(self.max_iter + 1):
            prev_decoder_out = prev_decoder_out._replace(
                step=step, max_step=self.max_iter + 1
            )
            decoder_out = model.forward_decoder(
                prev_decoder_out,
                encoder_out,
                eos_penalty=self.eos_penalty,
                max_ratio=self.max_ratio,
                decoding_format=self.decoding_format,
            )
            terminated, output_tokens, output_scores, output_attn = is_a_loop(
                self.pad,
                prev_output_tokens,
                decoder_out.output_tokens,
                decoder_out.output_scores,
                decoder_out.attn,
            )
            decoder_out = decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=output_scores,
                attn=output_attn,
            )

            terminated = last_step(step, self.max_iter, terminated)
            # collect finalized sentences
            finalized_idxs = sent_idxs[terminated]
            finalized_tokens = decoder_out.output_tokens[terminated]
            finalized_scores = decoder_out.output_scores[terminated]
            finalized_attn = (
                None if decoder_out.attn is None else decoder_out.attn[terminated]
            )
            finalized_tokens_list = finalize_hypos_loop_tokens(
                finalized_tokens_list,
                finalized_idxs,
                self.pad,
                finalized_tokens,
                finalized_scores,
            )
            finalized_scores_list = finalize_hypos_loop_scores(
                finalized_scores_list,
                finalized_idxs,
                self.pad,
                finalized_tokens,
                finalized_scores,
            )
            finalized_attns_list, finalized_alignments_list = finalize_hypos_loop_attns(
                finalized_attns_list,
                finalized_alignments_list,
                finalized_idxs,
                self.pad,
                finalized_tokens,
                finalized_scores,
                finalized_attn,
            )

            # for next step
            not_terminated = ~terminated
            prev_decoder_out = decoder_out._replace(
                output_tokens=script_skip_tensor(
                    decoder_out.output_tokens, not_terminated
                ),
                output_scores=script_skip_tensor(
                    decoder_out.output_scores, not_terminated
                ),
                attn=decoder_out.attn,
                step=decoder_out.step,
                max_step=decoder_out.max_step,
            )
            encoder_out = EncoderOut(
                encoder_out=script_skip_tensor(encoder_out.encoder_out, ~terminated),
                encoder_padding_mask=None,
                encoder_embedding=script_skip_tensor(
                    encoder_out.encoder_embedding, ~terminated
                ),
                encoder_states=None,
                src_tokens=None,
                src_lengths=None,
            )
            sent_idxs = script_skip_tensor(sent_idxs, not_terminated)

            prev_output_tokens = prev_decoder_out.output_tokens.clone()

        return (
            finalized_tokens_list,
            finalized_scores_list,
            finalized_attns_list,
            finalized_alignments_list,
        )
