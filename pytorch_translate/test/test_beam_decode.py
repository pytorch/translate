#!/usr/bin/env python3

import torch
import unittest

from fairseq import models
from pytorch_translate import beam_decode
from pytorch_translate import rnn  # noqa
from pytorch_translate.test import utils as test_utils


class TestBeamDecode(unittest.TestCase):
    @unittest.skipIf(
        torch.cuda.device_count() < 1, "No GPU available for test."
    )
    def test_basic_generate(self):
        test_args = test_utils.ModelParamsDict()
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        model = models.build_model(test_args, src_dict, tgt_dict)
        translator = beam_decode.SequenceGenerator([model])
        src_tokens = torch.LongTensor([[0, 0, 0], [0, 0, 0]])
        src_lengths = torch.LongTensor([3, 3])
        translator.generate(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )
