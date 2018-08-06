#!/usr/bin/env python3

from fairseq.tasks import register_task
from pytorch_translate.tasks import PytorchTranslateTask
from . import adversarial_criterion
from . import adversaries


@register_task("pytorch_translate_adversarial")
class PytorchTranslateAdversarialTask(PytorchTranslateTask):
    """Extends `PytorchTranslateTask` to account for the adversarial criterion
    and the adversary"""

    def build_adversary(self, args):
        return adversaries.build_adversary(args, self)

    def build_adversarial_criterion(self, args):
        return adversarial_criterion.build_criterion(args, self)
