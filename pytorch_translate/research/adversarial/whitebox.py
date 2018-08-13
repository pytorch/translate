#!/usr/bin/env python3

import collections
import os
from typing import NamedTuple

import torch
from fairseq import data, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from pytorch_translate import rnn  # noqa
from pytorch_translate import transformer  # noqa
from pytorch_translate import (
    options as pytorch_translate_options,
    utils as pytorch_translate_utils,
)
from pytorch_translate.research.adversarial import adversarial_criterion  # noqa
from pytorch_translate.research.adversarial import adversarial_tasks  # noqa
from pytorch_translate.research.adversarial import (
    adversarial_options,
    adversarial_trainer,
    adversaries,
)


class AttackInfo(NamedTuple):
    sample_id: int
    src_tokens: str
    target_tokens: str
    adv_tokens: str
    src_str: str
    target_str: str
    adv_str: str


def get_parser_with_args():
    """Create argument parser with arguments specific to this script"""
    parser = options.get_parser(
        "Whitebox attack", default_task="pytorch_translate_adversarial"
    )

    # Data related arguments
    data_group = pytorch_translate_options.add_dataset_args(parser, gen=True)

    # Adds args used by the standalone generate binary.
    data_group.add_argument(
        "--source-vocab-file",
        default="",
        metavar="FILE",
        help="Path to text file representing the Dictionary to use.",
    )
    data_group.add_argument(
        "--char-source-vocab-file",
        default="",
        metavar="FILE",
        help=(
            "Same as --source-vocab-file except using characters. "
            "(For use with char_source models only.)"
        ),
    )
    data_group.add_argument(
        "--target-vocab-file",
        default="",
        metavar="FILE",
        help="Path to text file representing the Dictionary to use.",
    )
    data_group.add_argument(
        "--source-text-file",
        default="",
        metavar="FILE",
        help="Path to raw text file containing examples in source dialect. "
        "This overrides what would be loaded from the data dir. ",
    )
    data_group.add_argument(
        "--target-text-file",
        default="",
        metavar="FILE",
        help="Path to raw text file containing examples in target dialect. "
        "This overrides what would be loaded from the data dir.",
    )
    data_group.add_argument(
        "--adversarial-output-file",
        default="",
        type=str,
        metavar="FILE",
        help="Path to text file to store the generated adversarial examples.",
    )

    # Adversarial attack specific group
    adversarial_options.add_adversarial_args(parser, attack_only=True)

    return parser


def validate_args(args):
    """Make sure the arguments are correct (files exist, no ensembles, etc...)"""
    # Verify that the path is specified and that it points to one model only
    assert args.path is not None, "--path required for generation!"
    assert (
        len(args.path.split(":")) == 1
    ), "Whitebox attacks on ensembles are not supported yet"
    # Check data files
    assert args.source_vocab_file and os.path.isfile(
        args.source_vocab_file
    ), "Please specify a valid file for --source-vocab-file"
    assert args.target_vocab_file and os.path.isfile(
        args.target_vocab_file
    ), "Please specify a valid file for --target-vocab_file"
    assert args.source_text_file and os.path.isfile(
        args.source_text_file
    ), "Please specify a valid file for --source-text-file"
    assert args.target_text_file and os.path.isfile(
        args.target_text_file
    ), "Please specify a valid file for --target-text-file"


def setup_attack(args):
    """Load model, data and create the AdversarialTrainer object"""

    # Setup task
    task = tasks.setup_task(args)

    # Load model
    models, models_args = pytorch_translate_utils.load_diverse_ensemble_for_inference(
        args.path.split(":"), task
    )

    # Only one model is supported as of now
    model, model_args = models[0], models_args[0]

    # Languages
    args.source_lang = model_args.source_lang
    args.target_lang = model_args.target_lang

    # Keep track of whether we reverse the source or not
    # (this is important to save the adversarial inputs in the correct order)
    args.reverse_source = model_args.reverse_source

    # Load dataset
    task.load_dataset_from_text(
        args.gen_subset,
        source_text_file=args.source_text_file,
        target_text_file=args.target_text_file,
        append_eos=model_args.append_eos_to_source,
        reverse_source=model_args.reverse_source,
    )

    # Create adversarial criterion
    adv_criterion = task.build_adversarial_criterion(args)

    # Adversary
    adversary = adversaries.build_adversary(args, model, task)

    # Print a bit of info
    print(
        f"| model {model_args.arch}, "
        f"adversarial criterion {adv_criterion.__class__.__name__}, "
        f"adversary {adversary.__class__.__name__}"
    )

    # Build trainer
    adv_trainer = adversarial_trainer.AdversarialTrainer(
        args=args,
        task=task,
        model=model,
        criterion=None,
        adversarial_criterion=adv_criterion,
        adversary=adversary,
    )

    # Device infos
    # For now only 1 GPU is supported
    distributed_world_size = getattr(args, "distributed_world_size", 1)
    print(f"| Attacking on {distributed_world_size} GPU(s)")
    print(
        f"| max tokens per GPU = {args.max_tokens} and \
        max sentences per GPU = {args.max_sentences}",
        flush=True,
    )

    return adv_trainer, task


def create_iterator(args, trainer, task, adv_split):
    """Sets up data and progress meters for one pass of adversarial attack."""
    # Set seed based on args.seed
    torch.manual_seed(args.seed)

    # reset training meters
    for k in ["wps", "ups", "wpb", "bsz"]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    return data.EpochBatchIterator(
        dataset=task.dataset(adv_split),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=trainer.get_model().max_positions(),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)


def _generate_adversarial_inputs(adv_trainer, args, task, adv_split):
    """Run the adversarial attack over the dataset"""

    # Keep track of the generated sentences
    # Initialize with empty translations
    adversarial_sentences = [""] * len(task.dataset(adv_split))

    # Initialize iterator
    itr = create_iterator(args, adv_trainer, task, adv_split)
    num_sentences = 0
    adversarial_samples = []
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        # Keep more detailed timing when invoked from benchmark
        if "keep_detailed_timing" in args:
            adv_timer = pytorch_translate_utils.BucketStopwatchMeter(
                args.increment, args.max_length, args.samples_per_length
            )
        else:
            adv_timer = StopwatchMeter()

        for attack_info in adversarial_attack_iterator(
            t, adv_trainer, task, adv_split, adv_timer, args.reverse_source
        ):
            if not args.quiet:
                print(f"S-{attack_info.sample_id}\t{attack_info.src_str}")
                print(f"A-{attack_info.sample_id}\t{attack_info.adv_str}")
            # Keep track of everything
            adversarial_sentences[attack_info.sample_id] = attack_info.adv_str
            adversarial_samples.append(
                collections.OrderedDict(
                    {
                        "sample_id": attack_info.sample_id,
                        "src_str": attack_info.src_str,
                        "target_str": attack_info.target_str,
                        "adv_str": attack_info.adv_str,
                    }
                )
            )
            wps_meter.update(attack_info.src_tokens.size(0))

            num_sentences += 1
            log_mid_attack_stats(t, adv_trainer)
    # If applicable, save the translations to the output file
    # For eg. external evaluation
    if getattr(args, "adversarial_output_file", False):
        with open(args.adversarial_output_file, "w") as out_file:
            for adv_str in adversarial_sentences:
                print(adv_str, file=out_file)

    return num_sentences, adv_timer, adversarial_samples


def adversarial_attack_iterator(
    itr, adv_trainer, task, adv_split, timer, reverse_source=False
):
    for sample in itr:
        net_input = sample["net_input"]
        bsz, srclen = net_input["src_tokens"].size()
        src_tokens = net_input["src_tokens"].long().cpu()
        # Generate the adversarial tokens
        timer.start()
        adv_tokens, _ = adv_trainer.gen_adversarial_examples(sample)
        adv_tokens = adv_tokens.long().cpu()
        timer.stop(srclen)
        # Iterate over the samples in the batch
        for b, id in enumerate(sample["id"]):
            # Remove padding
            nopad_src_tokens = utils.strip_pad(src_tokens[b], task.src_dict.pad())
            nopad_adv_tokens = utils.strip_pad(adv_tokens[b], task.src_dict.pad())
            # Retrieve source string
            src_str = task.dataset(adv_split).src.get_original_text(id)
            # Process the source string in reverse if applicable
            if reverse_source:
                src_str = " ".join(reversed(src_str.split()))
            # Convert to string
            adv_str = " ".join(
                # We do this to recover <unk> from the original source
                src_tok_str if src_tok == adv_tok else task.src_dict[adv_tok]
                for src_tok_str, src_tok, adv_tok in zip(
                    src_str.split(), nopad_src_tokens, nopad_adv_tokens
                )
            )
            # Reverse the string back if applicable
            if reverse_source:
                adv_str = " ".join(reversed(adv_str.split()))
            attack_info = AttackInfo(
                sample_id=id,
                src_tokens=net_input["src_tokens"][b].int().cpu(),
                target_tokens=sample["target"][b].int().cpu(),
                adv_tokens=adv_tokens[b],
                src_str=task.dataset(adv_split).src.get_original_text(id),
                target_str=task.dataset(adv_split).tgt.get_original_text(id),
                adv_str=adv_str,
            )
            yield attack_info


def log_mid_attack_stats(itr, adv_trainer):
    stats = get_attack_stats(adv_trainer)
    itr.log(stats)


def get_attack_stats(adv_trainer):
    stats = collections.OrderedDict()
    stats["wps"] = round(adv_trainer.get_meter("wps").avg)
    stats["wpb"] = round(adv_trainer.get_meter("wpb").avg)
    stats["bsz"] = round(adv_trainer.get_meter("bsz").avg)
    stats["oom"] = adv_trainer.get_meter("oom").avg
    return stats


def attack(args):
    print(args)

    adv_trainer, task = setup_attack(args)

    (num_sentences, gen_timer, adversarial_samples) = _generate_adversarial_inputs(
        adv_trainer=adv_trainer, args=args, task=task, adv_split=args.gen_subset
    )
    print(
        f"| Generated {num_sentences} adversarial inputs ({gen_timer.n} tokens) "
        f"in {gen_timer.sum:.1f}s ({1. / gen_timer.avg:.2f} tokens/s)"
    )


def main():
    parser = get_parser_with_args()
    args = adversarial_options.parse_args_and_adversary(parser)
    validate_args(args)
    attack(args)


if __name__ == "__main__":
    main()
