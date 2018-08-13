#!/usr/bin/env python3

import importlib
import os

from .base_adversary import BaseAdversary


ADVERSARY_REGISTRY = {}
ADVERSARY_CLASS_NAMES = set()


def build_adversary(args, model, task):
    return ADVERSARY_REGISTRY[args.adversary](args, model, task)


def register_adversary(name):
    """Decorator to register a new adversary."""

    def register_adversary_cls(cls):
        # Ensure the adversary registered names are unique
        if name in ADVERSARY_REGISTRY:
            raise ValueError(f"Cannot register duplicate adversary ({name})")
        # Ensure all adversaries inherit BaseAdversary
        if not issubclass(cls, BaseAdversary):
            raise ValueError(
                f"Adversary ({name}: {cls.__name__}) must extend BaseAdversary"
            )
        # Ensure the adversary class names are unique
        if cls.__name__ in ADVERSARY_CLASS_NAMES:
            raise ValueError(
                f"Cannot register adversary with duplicate class name ({cls.__name__})"
            )
        # Register the adversary
        ADVERSARY_REGISTRY[name] = cls
        ADVERSARY_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_adversary_cls


# automatically import any Python files in research/adversarial/adversaries/
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module(
            f"pytorch_translate.research.adversarial.adversaries.{module}"
        )
