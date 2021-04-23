#!/usr/bin/env python3

import importlib
import os

from pytorch_translate.attention.base_attention import BaseAttention


ATTENTION_REGISTRY = {}


def build_attention(attention_type, decoder_hidden_state_dim, context_dim, **kwargs):
    return ATTENTION_REGISTRY[attention_type](
        decoder_hidden_state_dim, context_dim, **kwargs
    )


def register_attention(name):
    """Decorator to register a new attention type."""

    def register_attention_cls(cls):
        if name in ATTENTION_REGISTRY:
            raise ValueError("Cannot register duplicate attention ({})".format(name))
        if not issubclass(cls, BaseAttention):
            raise ValueError(
                "Attention ({} : {}) must extend BaseAttention".format(
                    name, cls.__name__
                )
            )
        ATTENTION_REGISTRY[name] = cls
        return cls

    return register_attention_cls


# automatically import any Python files in the attention/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("pytorch_translate.attention.{}".format(module))
