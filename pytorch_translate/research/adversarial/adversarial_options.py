#!/usr/bin/env python3

import argparse

from fairseq.options import eval_str_list
from fairseq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY
from fairseq.criterions import CRITERION_REGISTRY
from fairseq.tasks import TASK_REGISTRY
from fairseq.optim import OPTIMIZER_REGISTRY
from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY
from pytorch_translate.research.adversarial.adversaries import (
    ADVERSARY_REGISTRY, BaseAdversary
)
from .adversarial_constraints import AdversarialConstraints


def add_adversarial_args(parser):
    """Adds arguments specific to adversarial example generation"""
    group = parser.add_argument_group("Adversarial examples arguments")

    group.add_argument("--quiet", action="store_true",
        help="Don't print the adversarial sentences to stdout")

    group.add_argument(
        "--path",
        metavar="DIR/FILE",
        default=None,
        help="path(s) to model file(s), colon separated "
        "(only one model is supported right now)"
    )
    # Trainer Arguments
    group.add_argument(
        "--modify-gradient",
        default="",
        metavar="ACTION",
        choices=["", "sign", "normalize", "normalize"],
        help="Modify the gradient by taking the sign or normalizing along the "
        "word vector dimension.",
    )
    # Adversarial criterion
    group.add_argument(
        "--adv-criterion",
        default="cross_entropy",
        metavar="CRIT",
        choices=CRITERION_REGISTRY.keys(),
        help="Adversarial criterion: {} (default: cross_entropy). "
        "This is the objective that the adversary will try to minimize. It "
        "should be something that will make the model worse at its training "
        "criterions".format(", ".join(CRITERION_REGISTRY.keys())),
    )
    group.add_argument(
        "--reverse-criterion",
        action="store_true",
        default=False,
        help="Force the adversary to *maximize* the criterion instead of "
        "minimizing it. This is convenient way of reusing training criterions "
        "for adversarial attacks.",
    )
    # Number of iterations for the attack
    group.add_argument(
        "--n-attack-iterations",
        default=1,
        metavar="N",
        type=int,
        help="Number of iterations during the attack. One iteration consists "
        "in: 1. Forward pass on the current input "
        "2. Backward pass to get the gradient"
        "3. Generate a new input with the adversary",
    )
    # Adversaries definitions can be found under research/adversarial/adversaries
    group.add_argument(
        "--adversary",
        default="random_swap",
        metavar="ADV",
        choices=ADVERSARY_REGISTRY.keys(),
        help="adversary type: {} (default: random_swap)".format(
            ", ".join(ADVERSARY_REGISTRY.keys())),
    )
    # Add arguments specific to all adversaries
    BaseAdversary.add_args(group)

    # Add constraints specific arguments
    AdversarialConstraints.add_args(parser)

    return group


def parse_args_and_adversary(parser, input_args=None):
    """This does the same thing as fairseq.options.parse_args_and_arch
    but for the criterion and adversary only"""
    # The parser doesn't know about adversary/criterion-specific args, so
    # we parse twice. First we parse the adversary/criterion, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)

    # Add model-specific args to parser.
    if hasattr(args, 'arch'):
        model_specific_group = parser.add_argument_group(
            'Model-specific configuration',
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        ARCH_MODEL_REGISTRY[args.arch].add_args(model_specific_group)

    # Add adversary-specific args to parser.
    adversary_specific_group = parser.add_argument_group(
        f"Arguments for adversary \"{args.adversary}\"",
        # Only include attributes which are explicitly given as command-line
        # arguments or which have default values.
        argument_default=argparse.SUPPRESS,
    )
    ADVERSARY_REGISTRY[args.adversary].add_args(adversary_specific_group)

    # Add adversarial criterion-specific args to parser.
    adv_criterion_specific_group = parser.add_argument_group(
        f"Arguments for adversarial criterion \"{args.adv_criterion}\"",
        # Only include attributes which are explicitly given as command-line
        # arguments or which have default values.
        argument_default=argparse.SUPPRESS,
    )
    CRITERION_REGISTRY[args.adv_criterion].add_args(adv_criterion_specific_group)

    if hasattr(args, 'criterion'):
        # Add criterion-specific args to parser.
        criterion_specific_group = parser.add_argument_group(
            f"Arguments for criterion \"{args.criterion}\"",
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        CRITERION_REGISTRY[args.criterion].add_args(criterion_specific_group)

    # Add other *-specific args to parser.
    if hasattr(args, 'optimizer'):
        OPTIMIZER_REGISTRY[args.optimizer].add_args(parser)
    if hasattr(args, 'lr_scheduler'):
        LR_SCHEDULER_REGISTRY[args.lr_scheduler].add_args(parser)
    if hasattr(args, 'task'):
        TASK_REGISTRY[args.task].add_args(parser)

    # Parse a second time.
    args = parser.parse_args(input_args)

    # Post-process args.
    if hasattr(args, 'lr'):
        args.lr = eval_str_list(args.lr, type=float)
    if hasattr(args, 'update_freq'):
        args.update_freq = eval_str_list(args.update_freq, type=int)
    if hasattr(args, 'max_sentences_valid') and args.max_sentences_valid is None:
        args.max_sentences_valid = args.max_sentences
    # The following line is a hack to be able to use the cross_entropy
    # criterion without polluting the command line with unnecessary arguments
    if not hasattr(args, "sentence_avg"):
        args.sentence_avg = False
    # this is another hack to ignore the multilingual case
    if not hasattr(args, "multiling_source_lang"):
        args.multiling_source_lang = None

    # Apply architecture configuration.
    if hasattr(args, 'arch'):
        ARCH_CONFIG_REGISTRY[args.arch](args)

    return args
