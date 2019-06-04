#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    base_architecture as transformer_base_architecture,
)
from fairseq.models.transformer_from_pretrained_xlm import (
    TransformerFromPretrainedXLMModel,
)
from pytorch_translate.data.masked_lm_dictionary import MaskedLMDictionary


@register_model("pytorch_translate_transformer_from_pretrained_xlm")
class PytorchTranslateTransformerFromPretrainedXLMModel(
    TransformerFromPretrainedXLMModel
):
    @classmethod
    def build_model(cls, args, task):
        return super().build_model(args, task, cls_dictionary=MaskedLMDictionary)


@register_model_architecture(
    "pytorch_translate_transformer_from_pretrained_xlm",
    "pytorch_translate_transformer_from_pretrained_xlm",
)
def base_architecture(args):
    transformer_base_architecture(args)
