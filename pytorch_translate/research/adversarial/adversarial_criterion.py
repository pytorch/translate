#!/usr/bin/env python3

from fairseq.criterions import CRITERION_REGISTRY


def build_criterion(args, task):
    """Same as fairseq.criterions.build_criterion but for adversarial criterion"""
    return CRITERION_REGISTRY[args.adv_criterion](args, task)
