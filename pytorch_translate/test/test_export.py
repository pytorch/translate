#!/usr/bin/env python3

import io
import logging
import unittest

import numpy as np
import torch
from pytorch_translate import char_source_hybrid  # noqa
from pytorch_translate import char_source_model  # noqa
from pytorch_translate import char_source_transformer_model  # noqa
from pytorch_translate import rnn  # noqa
from pytorch_translate import transformer  # noqa
from pytorch_translate import constants
from pytorch_translate.ensemble_export import BeamSearch, BeamSearchAndDecode
from pytorch_translate.research.knowledge_distillation import (  # noqa
    dual_decoder_kd_model,
    hybrid_dual_decoder_kd_model,
)
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


logger = logging.getLogger(__name__)


class TestPyTorchExport(unittest.TestCase):
    def _test_full_beam_decoder(self, test_args, quantize=False):
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        sample = next(samples)
        # [seq len, batch size=1]
        src_tokens = sample["net_input"]["src_tokens"][0:1].t()
        # [seq len]
        src_lengths = sample["net_input"]["src_lengths"][0:1].long()

        num_models = 3
        model_list = []
        for _ in range(num_models):
            model_list.append(task.build_model(test_args))

        length, word_length = 11, 7
        if test_args.arch in constants.ARCHS_FOR_CHAR_SOURCE:
            char_inds = torch.LongTensor(
                np.random.randint(0, 126, (1, length, word_length), dtype="int64")
            )
            word_lengths = torch.IntTensor(
                np.array([word_length] * length, dtype="int32")
            ).reshape((1, length))
        else:
            char_inds, word_lengths = None, None

        beam_size = 6
        bs = BeamSearch(
            model_list,
            tgt_dict,
            src_tokens,
            src_lengths,
            beam_size=beam_size,
            quantize=quantize,
            char_inds=char_inds,
            word_lengths=word_lengths,
        )
        f = io.BytesIO()
        bs.save_to_pytorch(f)

        # Test generalization with a different sequence length
        src_tokens = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).unsqueeze(1)
        src_lengths = torch.LongTensor([11])
        prev_token = torch.LongTensor([0])
        prev_scores = torch.FloatTensor([0.0])
        attn_weights = torch.zeros(src_tokens.shape[0])
        prev_hypos_indices = torch.zeros(beam_size, dtype=torch.int64)

        outs = bs(
            src_tokens,
            src_lengths,
            prev_token,
            prev_scores,
            attn_weights,
            prev_hypos_indices,
            torch.LongTensor([20]),
            char_inds=char_inds,
            word_lengths=word_lengths,
        )

        f.seek(0)
        deserialized_bs = torch.jit.load(f)
        deserialized_bs.apply(lambda s: s._unpack() if hasattr(s, "_unpack") else None)
        outs_deserialized = deserialized_bs(
            src_tokens,
            src_lengths,
            prev_token,
            prev_scores,
            attn_weights,
            prev_hypos_indices,
            torch.LongTensor([20]),
            char_inds=char_inds,
            word_lengths=word_lengths,
        )

        for a, b in zip(outs_deserialized, outs):
            np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy())

    def test_full_beam_decoder(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        self._test_full_beam_decoder(test_args)

    def test_full_beam_decoder_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 10,
            "max_translation_candidates_per_word": 1,
        }
        self._test_full_beam_decoder(test_args)

    def test_full_beam_decoder_char_rnn_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 10,
            "max_translation_candidates_per_word": 1,
        }

        test_args.arch = "char_source"
        test_args.char_source_dict_size = 126
        test_args.char_embed_dim = 8
        test_args.char_rnn_units = 12
        test_args.char_rnn_layers = 2

        self._test_full_beam_decoder(test_args)

    def test_full_beam_decoder_char_cnn_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 10,
            "max_translation_candidates_per_word": 1,
        }

        test_args.arch = "char_source"
        test_args.char_source_dict_size = 126
        test_args.char_embed_dim = 8
        test_args.char_cnn_params = "[(10, 3), (10, 5)]"
        test_args.char_cnn_nonlinear_fn = "tanh"
        test_args.char_cnn_pool_type = "max"
        test_args.char_cnn_num_highway_layers = 2

        self._test_full_beam_decoder(test_args)

    def test_full_beam_decoder_aan(self):
        test_args = test_utils.ModelParamsDict(arch="transformer")
        test_args.aan = True
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 10,
            "max_translation_candidates_per_word": 1,
        }
        self._test_full_beam_decoder(test_args, quantize=True)

    def test_full_beam_decoder_aan_bottlenceck(self):
        test_args = test_utils.ModelParamsDict(arch="transformer")
        test_args.aan = True
        test_args.decoder_out_embed_dim = 5
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 10,
            "max_translation_candidates_per_word": 1,
        }
        self._test_full_beam_decoder(test_args, quantize=True)

    def test_full_beam_decoder_char_transformer(self):
        test_args = test_utils.ModelParamsDict(arch="char_transformer")
        test_args.char_source_dict_size = 126
        test_args.char_embed_dim = 8
        test_args.char_cnn_params = "[(10, 3), (10, 5)]"
        test_args.char_cnn_nonlinear_fn = "tanh"
        test_args.char_cnn_pool_type = "max"
        test_args.char_cnn_num_highway_layers = 2

        self._test_full_beam_decoder(test_args)

    def test_full_beam_decoder_char_source_hybrid(self):
        test_args = test_utils.ModelParamsDict(arch="char_source_hybrid")
        test_args.char_source_dict_size = 126
        test_args.char_embed_dim = 8
        test_args.char_cnn_params = "[(8, 3), (8, 5)]"
        test_args.char_cnn_nonlinear_fn = "tanh"
        test_args.char_cnn_pool_type = "max"
        test_args.char_cnn_num_highway_layers = 2
        self._test_full_beam_decoder(test_args)


class TestBeamSearchAndDecodeExport(unittest.TestCase):
    def _test_full_beam_search_decoder(self, test_args, quantize=False):
        samples, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        sample = next(samples)
        # [seq len, batch size=1]
        src_tokens = sample["net_input"]["src_tokens"][0:1].t()
        # [seq len]
        src_lengths = sample["net_input"]["src_lengths"][0:1].long()

        num_models = 3
        model_list = []
        for _ in range(num_models):
            model_list.append(task.build_model(test_args))

        eos_token_id = 8
        length_penalty = 0.25
        nbest = 3
        stop_at_eos = True
        num_steps = torch.LongTensor([20])

        beam_size = 6
        bsd = BeamSearchAndDecode(
            model_list,
            tgt_dict,
            src_tokens,
            src_lengths,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            nbest=nbest,
            beam_size=beam_size,
            stop_at_eos=stop_at_eos,
            quantize=quantize,
        )
        f = io.BytesIO()
        bsd.save_to_pytorch(f)

        # Test generalization with a different sequence length
        src_tokens = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11]).unsqueeze(1)
        src_lengths = torch.LongTensor([11])
        prev_token = torch.LongTensor([0])
        prev_scores = torch.FloatTensor([0.0])
        attn_weights = torch.zeros(src_tokens.shape[0])
        prev_hypos_indices = torch.zeros(beam_size, dtype=torch.int64)

        outs = bsd(
            src_tokens,
            src_lengths,
            prev_token,
            prev_scores,
            attn_weights,
            prev_hypos_indices,
            num_steps[0],
        )

        f.seek(0)
        deserialized_bsd = torch.jit.load(f)
        deserialized_bsd.apply(lambda s: s._unpack() if hasattr(s, "_unpack") else None)
        outs_deserialized = deserialized_bsd(
            src_tokens,
            src_lengths,
            prev_token,
            prev_scores,
            attn_weights,
            prev_hypos_indices,
            num_steps[0],
        )

        for hypo, hypo_deserialized in zip(outs, outs_deserialized):
            np.testing.assert_array_equal(
                hypo[0].tolist(), hypo_deserialized[0].tolist()
            )
            np.testing.assert_array_almost_equal(
                hypo[2], hypo_deserialized[2], decimal=1
            )
            np.testing.assert_array_almost_equal(
                hypo[3].numpy(), hypo_deserialized[3].numpy(), decimal=1
            )

    def test_full_beam_search_decoder(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        self._test_full_beam_search_decoder(test_args)

    def test_full_beam_search_decoder_reverse(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True, reverse_source=True
        )
        self._test_full_beam_search_decoder(test_args)

    def test_full_beam_search_decoder_vocab_reduction(self):
        test_args = test_utils.ModelParamsDict(
            encoder_bidirectional=True, sequence_lstm=True
        )
        lexical_dictionaries = test_utils.create_lexical_dictionaries()
        test_args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": 10,
            "max_translation_candidates_per_word": 1,
        }
        self._test_full_beam_search_decoder(test_args)

    def test_full_beam_search_decoder_hybrid(self):
        test_args = test_utils.ModelParamsDict(arch="hybrid_transformer_rnn")
        self._test_full_beam_search_decoder(test_args)
