#!/usr/bin/env python3

import numpy as np
import torch
import unittest

from pytorch_translate import beam_decode
from pytorch_translate import rnn  # noqa
from pytorch_translate import char_source_model  # noqa (must be after rnn)
from pytorch_translate import tasks
from pytorch_translate.test import utils as test_utils


class TestBeamDecode(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_basic_generate(self):
        test_args = test_utils.ModelParamsDict()
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        model = task.build_model(test_args)
        translator = beam_decode.SequenceGenerator([model], task.target_dictionary)
        src_tokens = torch.LongTensor([[0, 0, 0], [0, 0, 0]])
        src_lengths = torch.LongTensor([3, 3])
        encoder_input = (src_tokens, src_lengths)
        translator.generate(encoder_input, maxlen=7)

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_char_rnn_generate(self):
        test_args = test_utils.ModelParamsDict(sequence_lstm=True)
        test_args.arch = "char_source"
        test_args.char_source_dict_size = 126
        test_args.char_embed_dim = 8
        test_args.char_rnn_units = 12
        test_args.char_rnn_layers = 2

        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        model = task.build_model(test_args)
        translator = beam_decode.SequenceGenerator([model], task.target_dictionary)
        src_tokens = torch.LongTensor([[0, 0, 0], [0, 0, 0]])
        src_lengths = torch.LongTensor([3, 3])
        char_inds = torch.LongTensor(np.zeros((2, 3, 5)))
        word_lengths = torch.LongTensor([[5, 5, 5], [5, 5, 5]])
        encoder_input = (src_tokens, src_lengths, char_inds, word_lengths)
        translator.generate(encoder_input, maxlen=7)
