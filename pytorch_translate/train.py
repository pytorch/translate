#!/usr/bin/env python3

import argparse
import collections
import itertools
import math
import multiprocessing
import os
import random
import shutil
import signal
import tempfile
import time
import torch

from typing import Any, Dict, Optional, Tuple

from fairseq import criterions, distributed_utils, models, options, progress_bar
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.trainer import Trainer

from pytorch_translate import average_checkpoints
from pytorch_translate import data as pytorch_translate_data
from pytorch_translate import generate
from pytorch_translate import rnn  # noqa
from pytorch_translate.research.word_prediction import word_prediction_criterion  # noqa
from pytorch_translate.research.word_prediction import word_prediction_model  # noqa


def get_parser_with_args():
    parser = options.get_parser("Trainer")
    options.add_dataset_args(parser, train=True, gen=True)
    options.add_distributed_training_args(parser)
    options.add_optimization_args(parser)
    options.add_checkpoint_args(parser)
    options.add_model_args(parser)
    options.add_generation_args(parser)

    parser.add_argument(
        "--log-verbose",
        action="store_true",
        help="Whether to output more verbose logs for debugging/profiling.",
    )

    # Adds args related to training (validation and stopping criterions).
    group = parser.add_argument_group("Optimization")
    group.add_argument(
        "--subepoch-validate-interval",
        default=0,
        type=int,
        metavar="N",
        help="Calculates loss over the validation set every N batch updates. "
        "Note that validation is done at the end of every epoch regardless. "
        "A value of <= 0 disables this.",
    )
    group.add_argument(
        "--stop-time-hr",
        default=-1,
        type=int,
        metavar="N",
        help="Stops training after N hours have elapsed. "
        "A value of < 0 disables this.",
    )
    group.add_argument(
        "--stop-no-best-validate-loss",
        default=-1,
        type=int,
        metavar="N",
        help="Stops training after N validations have been run without "
        "achieving a better loss than before. Note that this is affected by "
        "--validation-interval in how frequently we run validation in the "
        "first place. A value of < 0 disables this.",
    )
    group.add_argument(
        "--stop-no-best-bleu-eval",
        default=-1,
        type=int,
        metavar="N",
        help="Stops training after N evals have been run without "
        "achieving a better BLEU score than before. Note that this is affected "
        "by --generate-bleu-eval-interval in how frequently we run BLEU eval "
        "in the first place. A value of < 0 disables this.",
    )

    # Args related to dataset.
    group = parser.add_argument_group("Dataset and data loading")
    group.add_argument(
        "--source-vocab-file",
        default="",
        metavar="FILE",
        help="Path to text file representing the fairseq Dictionary to use. "
        "If left empty, the dict is auto-generated from source training data.",
    )
    group.add_argument(
        "--source-max-vocab-size",
        default=-1,
        type=int,
        metavar="N",
        help="If a new vocab file needs to be generated, restrict it to the "
        "top N most common words. If we re-use an existing vocab file, this "
        "flag will have no effect. A value of < 0 means no max size.",
    )
    group.add_argument(
        "--target-vocab-file",
        default="",
        metavar="FILE",
        help="Path to text file representing the fairseq Dictionary to use. "
        "If left empty, the dict is auto-generated from target training data.",
    )
    group.add_argument(
        "--target-max-vocab-size",
        default=-1,
        type=int,
        metavar="N",
        help="If a new vocab file needs to be generated, restrict it to the "
        "top N most common words. If we re-use an existing vocab file, this "
        "flag will have no effect. A value of < 0 means no max size.",
    )
    group.add_argument(
        "--train-source-text-file",
        default="",
        metavar="FILE",
        help="Path to raw text file containing source training examples. "
        "This overrides what would be loaded from the data dir.",
    )
    group.add_argument(
        "--train-target-text-file",
        default="",
        metavar="FILE",
        help="Path to raw text file containing target training examples. "
        "This overrides what would be loaded from the data dir.",
    )
    group.add_argument(
        "--eval-source-text-file",
        default="",
        metavar="FILE",
        help="Path to raw text file containing source eval examples for "
        "calculating validation loss and BLEU eval scores. "
        "This overrides what would be loaded from the data dir.",
    )
    group.add_argument(
        "--eval-target-text-file",
        default="",
        metavar="FILE",
        help="Path to raw text file containing target eval examples for "
        "calculating validation loss and BLEU eval scores. "
        "This overrides what would be loaded from the data dir.",
    )
    group.add_argument(
        "--penalized-target-tokens-file",
        default="",
        metavar="FILE",
        help="Path to text file of tokens to receive a penalty in decoding."
        "If left empty, no penalty will be applied",
    )

    # Adds args related to checkpointing.
    group = parser.add_argument_group("Checkpointing")
    group.add_argument(
        "--no-end-of-epoch-checkpoints",
        action="store_true",
        help="Disables saving checkpoints at the end of the epoch. "
        "This differs from --no-save and --no-epoch-checkpoints in that it "
        "still allows for intra-epoch checkpoints if --save-interval is set.",
    )
    group.add_argument(
        "--max-checkpoints-kept",
        default=-1,
        type=int,
        metavar="N",
        help="Keep at most the last N checkpoints file around. "
        "A value < -1 keeps all. "
        "When --generate-bleu-eval-avg-checkpoints is used and is > N, the "
        "number of checkpoints kept around is automatically adjusted "
        "to allow BLEU to work properly.",
    )

    # Adds args for generating intermediate BLEU eval while training.
    # generate.add_args() adds args used by both train.py and the standalone
    # generate binary, while the flags defined here are used only by train.py.
    generate.add_args(parser)
    group = parser.add_argument_group("Generation")
    group.add_argument(
        "--generate-bleu-eval-per-epoch",
        action="store_true",
        help="Whether to generate BLEU score eval after each epoch.",
    )
    group.add_argument(
        "--generate-bleu-eval-interval",
        default=0,
        type=int,
        metavar="N",
        help="Does BLEU eval every N batch updates. Note that "
        "--save-interval also affects this - we can only eval as "
        "frequently as a checkpoint is written. A value of <= 0 "
        "disables this.",
    )
    group.add_argument(
        "--generate-bleu-eval-avg-checkpoints",
        default=1,
        type=int,
        metavar="N",
        help="Maximum number of last N checkpoints to average over when "
        "doing BLEU eval. Must be >= 1.",
    )
    group.add_argument(
        "--continuous-averaging-after-epochs",
        type=int,
        default=-1,
        help=(
            "Average parameter values after each step since previous "
            "checkpoint, beginning after the specified number of epochs. "
        ),
    )

    return parser


def parse_args_and_arch(parser):
    args = options.parse_args_and_arch(parser)
    # Prevents generate from printing individual translated sentences when
    # calculating BLEU score.
    args.quiet = True

    assert_corpora_files_specified(args)

    print(args)
    return args


def load_existing_checkpoint(save_dir, restore_file, trainer):
    # Load the latest checkpoint if one is available
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, restore_file)
    extra_state = trainer.load_checkpoint(checkpoint_path)
    if extra_state is not None:
        print(
            f"| loaded checkpoint {checkpoint_path} (epoch {extra_state['epoch']})\n"
            f"| extra_state {extra_state}"
        )
        # batch_offset being None denotes this was a checkpoint saved at
        # the end of an epoch (after the last batch).
        if extra_state["batch_offset"] is None:
            trainer.lr_step(extra_state["epoch"])
            extra_state["epoch"] += 1
            extra_state["batch_offset"] = 0

        # check availability for checkpoint backward compatiblity
        if "start_time" not in extra_state:
            extra_state["start_time"] = time.time()

        if "last_bleu_eval" not in extra_state:
            extra_state["last_bleu_eval"] = 0

    else:
        extra_state = {
            "epoch": 1,
            "batch_offset": 0,
            "val_loss": None,
            "start_time": time.time(),
            "last_bleu_eval": 0,
        }
    return extra_state


def assert_corpora_files_specified(args):
    assert not args.data, (
        "Specifying a data directory is disabled in FBTranslate since the "
        "fairseq data class is not supported. Please specify "
        "--train-source-text-file, --train-target-text-file, "
        "--eval-source-text-file, and  --eval-target-text-file instead."
    )
    assert (
        args.train_source_text_file and os.path.isfile(args.train_source_text_file)
    ), "Please specify a valid file for --train-source-text-file"
    assert (
        args.train_target_text_file and os.path.isfile(args.train_target_text_file)
    ), "Please specify a valid file for --train-target-text-file"
    assert (
        args.eval_source_text_file and os.path.isfile(args.eval_source_text_file)
    ), "Please specify a valid file for --eval-source-text-file"
    assert (
        args.eval_target_text_file and os.path.isfile(args.eval_target_text_file)
    ), "Please specify a valid file for --eval-target-text-file"


def setup_training(args):
    """Parse args, load dataset, and load model trainer."""
    if not torch.cuda.is_available():
        raise NotImplementedError("Training on CPU is not supported")
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Load dataset
    splits = [args.train_subset, args.valid_subset]

    if args.source_lang is None:
        args.source_lang = "src"
    if args.target_lang is None:
        args.target_lang = "tgt"

    assert_corpora_files_specified(args)
    train_corpus = pytorch_translate_data.ParallelCorpusConfig(
        source=pytorch_translate_data.CorpusConfig(
            dialect=args.source_lang, data_file=args.train_source_text_file
        ),
        target=pytorch_translate_data.CorpusConfig(
            dialect=args.target_lang, data_file=args.train_target_text_file
        ),
    )
    eval_corpus = pytorch_translate_data.ParallelCorpusConfig(
        source=pytorch_translate_data.CorpusConfig(
            dialect=args.source_lang, data_file=args.eval_source_text_file
        ),
        target=pytorch_translate_data.CorpusConfig(
            dialect=args.target_lang, data_file=args.eval_target_text_file
        ),
    )

    if args.log_verbose:
        print("Starting to load raw text files.", flush=True)
    dataset = pytorch_translate_data.load_raw_text_dataset(
        train_corpus=train_corpus,
        eval_corpus=eval_corpus,
        train_split=args.train_subset,
        eval_split=args.valid_subset,
        args=args,
    )
    if args.log_verbose:
        print("Finished loading dataset", flush=True)
    if args.source_lang is None or args.target_lang is None:
        # record inferred languages in args, so that it's saved in checkpoints
        args.source_lang, args.target_lang = dataset.src, dataset.dst

    print(f"| [{dataset.src}] dictionary: {len(dataset.src_dict)} types")
    print(f"| [{dataset.dst}] dictionary: {len(dataset.dst_dict)} types")

    for split in splits:
        print(f"| {split} {len(dataset.splits[split])} examples")

    # Build model and criterion
    model = models.build_model(args, dataset.src_dict, dataset.dst_dict)
    criterion = criterions.build_criterion(args, dataset.src_dict, dataset.dst_dict)
    print(f"| model {args.arch}, criterion {criterion.__class__.__name__}")
    print(
        f"| num. model params: \
        {sum(p.data.numel() for p in model.parameters())}"
    )

    # Build trainer
    trainer = Trainer(args, model, criterion)
    print(f"| training on {args.distributed_world_size} GPUs")
    print(
        f"| max tokens per GPU = {args.max_tokens} and \
        max sentences per GPU = {args.max_sentences}",
        flush=True,
    )

    extra_state = load_existing_checkpoint(args.save_dir, args.restore_file, trainer)

    return extra_state, trainer, dataset


def single_process_main(args):
    """Train the model for multiple epochs."""
    extra_state, trainer, dataset = setup_training(args)

    train_iterator = train(
        args=args, extra_state=extra_state, trainer=trainer, dataset=dataset
    )

    for _ in train_iterator:
        pass


def train(args, extra_state, trainer, dataset):
    # offset for current epoch (may be different from checkpoint offset)
    starting_offset = extra_state["batch_offset"]

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    while lr > args.min_lr and extra_state["epoch"] <= max_epoch:
        """Train the model for one epoch."""

        itr, progress, extra_meters = setup_epoch(
            args=args,
            epoch=extra_state["epoch"],
            batch_offset=starting_offset,
            trainer=trainer,
            dataset=dataset,
        )

        for i, sample in enumerate(itr, start=starting_offset):
            log_output = trainer.train_step(sample)

            train_stats = log_mid_epoch_stats(
                trainer=trainer,
                progress=progress,
                extra_meters=extra_meters,
                log_output=log_output,
            )

            if (
                args.continuous_averaging_after_epochs >= 0
                and extra_state["epoch"] > args.continuous_averaging_after_epochs
            ):
                model_param_dict = trainer.model.state_dict()
                if "param_totals" not in extra_state:
                    extra_state["param_totals"] = {}
                    for name, value in model_param_dict.items():
                        extra_state["param_totals"][name] = value.clone()
                    extra_state["param_accum_count"] = 1
                else:
                    for name, value in model_param_dict.items():
                        extra_state["param_totals"][name] += value
                    extra_state["param_accum_count"] += 1

            if i == starting_offset:
                # ignore the first mini-batch in words-per-second calculation
                trainer.get_meter("wps").reset()

            num_updates = trainer.get_num_updates()
            do_validate = (
                args.subepoch_validate_interval > 0
                and num_updates % args.subepoch_validate_interval == 0
            )
            do_save = (
                not args.no_save
                and args.save_interval > 0
                and num_updates % args.save_interval == 0
            )
            do_eval_bleu = (
                # We can only do BLEU eval when we have a new checkpoint to load.
                do_save
                and args.generate_bleu_eval_interval > 0
                and num_updates - extra_state["last_bleu_eval"]
                >= args.generate_bleu_eval_interval
            )
            if do_eval_bleu:
                extra_state["last_bleu_eval"] = num_updates

            extra_state["batch_offset"] = i + 1

            (
                _, val_ppl, val_bleu, stop_training_mid_epoch
            ) = validate_save_and_evaluate_bleu(
                args=args,
                trainer=trainer,
                dataset=dataset,
                extra_state=extra_state,
                do_validate=do_validate,
                do_save=do_save,
                do_eval_bleu=do_eval_bleu,
            )
            yield (
                trainer.get_num_updates(),
                {
                    "train_ppl": train_stats["ppl"],
                    "tune_ppl": val_ppl,
                    "tune_bleu": val_bleu,
                },
            )

            if stop_training_mid_epoch:
                break

        # log end-of-epoch stats
        train_stats = log_end_epoch_stats(
            trainer=trainer, progress=progress, extra_meters=extra_meters
        )

        if stop_training_mid_epoch:
            break

        # batch_offset being None denotes the end of an epoch.
        extra_state["batch_offset"] = None
        (
            val_loss, val_ppl, val_bleu, stop_training_end_of_epoch
        ) = validate_save_and_evaluate_bleu(
            args=args,
            trainer=trainer,
            dataset=dataset,
            extra_state=extra_state,
            do_validate=True,
            do_save=not args.no_save and not args.no_end_of_epoch_checkpoints,
            do_eval_bleu=args.generate_bleu_eval_per_epoch,
        )
        extra_state["val_loss"] = val_loss
        yield (
            trainer.get_num_updates(),
            {
                "train_ppl": train_stats["ppl"],
                "tune_ppl": val_ppl,
                "tune_bleu": val_bleu,
            },
        )
        if stop_training_end_of_epoch:
            break

        lr = trainer.lr_step(extra_state["epoch"], val_loss)
        extra_state["epoch"] += 1
        extra_state["batch_offset"] = 0
        starting_offset = 0

        if is_training_over_time_limit(extra_state["start_time"], args.stop_time_hr):
            break

    train_meter.stop()
    print(f"| done training in {train_meter.sum:.1f} seconds")

    if "evaluate_bleu" in extra_state:
        print(
            f"| Best BLEU score of {extra_state['evaluate_bleu']['best']} was from "
            f"epoch {extra_state['evaluate_bleu']['best_epoch']}"
        )


def is_training_over_time_limit(start_time, stop_time):
    elapsed_hr = (time.time() - start_time) / (60 * 60)
    training_over_time_limit = False
    if stop_time >= 0 and elapsed_hr > stop_time:
        print(
            f"Stopping training due to stop time limit - it has been  "
            f"{elapsed_hr} hours since starting training at {start_time}."
        )
        training_over_time_limit = True
    return training_over_time_limit


def get_perplexity(loss):
    try:
        return f"{math.pow(2, loss):.2f}"
    except OverflowError:
        return float("inf")


def setup_epoch(args, epoch, batch_offset, trainer, dataset):
    """Sets up data and progress meters for one epoch."""
    # Set seed based on args.seed and the epoch number so that we get
    # reproducible results when resuming from checkpoints
    seed = args.seed + epoch
    torch.manual_seed(seed)

    # The max number of positions can be different for train and valid
    # e.g., RNNs may support more positions at test time than seen in training
    max_positions_train = (
        min(args.max_source_positions, trainer.get_model().max_encoder_positions()),
        min(args.max_target_positions, trainer.get_model().max_decoder_positions()),
    )

    # Initialize dataloader, starting at batch_offset
    itr = dataset.train_dataloader(
        args.train_subset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions_train,
        seed=seed,
        epoch=epoch,
        sample_without_replacement=args.sample_without_replacement,
        sort_by_source_size=(epoch <= args.curriculum),
        shard_id=args.distributed_rank,
        num_shards=args.distributed_world_size,
    )
    progress = progress_bar.build_progress_bar(
        args, itr, epoch, no_progress_bar="simple"
    )
    itr = itertools.islice(progress, batch_offset, None)

    # reset training meters
    for k in ["train_loss", "train_nll_loss", "wps", "ups", "wpb", "bsz", "clip"]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    return itr, progress, extra_meters


def log_mid_epoch_stats(trainer, progress, extra_meters, log_output):
    stats = get_training_stats(trainer)
    for k, v in log_output.items():
        if k in ["loss", "nll_loss"]:
            continue  # these are already logged above
        if "loss" in k:
            extra_meters[k].update(v, log_output["sample_size"])
        else:
            extra_meters[k].update(v)
        stats[k] = extra_meters[k].avg
    progress.log(stats)
    return stats


def log_end_epoch_stats(trainer, progress, extra_meters):
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)
    return stats


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats["loss"] = f"{trainer.get_meter('train_loss').avg:.3f}"
    if trainer.get_meter("train_nll_loss").count > 0:
        nll_loss = trainer.get_meter("train_nll_loss").avg
        stats["nll_loss"] = f"{nll_loss:.3f}"
    else:
        nll_loss = trainer.get_meter("train_loss").avg
    stats["ppl"] = get_perplexity(nll_loss)
    stats["wps"] = round(trainer.get_meter("wps").avg)
    stats["ups"] = f"{trainer.get_meter('ups').avg:.1f}"
    stats["wpb"] = round(trainer.get_meter("wpb").avg)
    stats["bsz"] = round(trainer.get_meter("bsz").avg)
    stats["num_updates"] = trainer.get_num_updates()
    stats["lr"] = trainer.get_lr()
    stats["gnorm"] = f"{trainer.get_meter('gnorm').avg:.3f}"
    stats["clip"] = f"{trainer.get_meter('clip').avg:.0%}"
    stats["oom"] = trainer.get_meter("oom").avg
    return stats


def save_checkpoint_maybe_continuous(filename, trainer, extra_state):
    if "param_totals" not in extra_state:
        trainer.save_checkpoint(filename, extra_state)
        return

    # trainer.save_checkpoint generates the structure used to save checkpoints
    with tempfile.TemporaryFile() as buffer:
        trainer.save_checkpoint(buffer, extra_state)
        buffer.seek(0)
        state = torch.load(
            buffer,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )
        buffer.close()

    param_accum_count = extra_state["param_accum_count"]
    for param_name, param_value in extra_state["param_totals"].items():
        state["model"][param_name] = param_value / param_accum_count
    torch.save(state, filename)

    # begin averaging anew after saving checkpoint
    extra_state.pop("param_totals")


# The purpose of this class is to keep track of the list of checkpoints
# currently alive and automatically delete those that are no more required
# and that we do not want to keep around.
# In a nutshell, this class remembers the last max_num_checkpoints
# and delete (auto_clear == True) the oldest checkpoint each time a new one
# is added past this number.
class ManagedCheckpoints:

    # - max_num_checkpoints: Maximum number of checkpoints we need at one point.
    # - auto_clear: Control whether or not checkpoints should get deleted when
    #   they are not in the last max_num_checkpoints appended to the
    #   self anymore.
    def __init__(self, max_num_checkpoints, auto_clear):
        self.auto_clear = auto_clear
        assert max_num_checkpoints > 0, "Empty listing is not supported"
        self.kept_checkpoints = collections.deque(maxlen=max_num_checkpoints)

    def append(self, checkpoint_filename):
        # If we append a filename that we already manage, we would need
        # to remove it from its current position otherwise it may get deleted
        # by the time we reach the use for this append.
        # E.g., Let us assume we have a max of 2 checkpoint.
        # We insert last_checkpoint, use it, then insert last_checkpoint,
        # use it, then insert it again. The first file gets delete, but it
        # is actually the same as the current one, so we actually delete
        # the current one. Then we try to use it and we will get an error
        # for file not found.
        # Although this is pretty easy to support this case, given we only
        # append the same file names with no_epoch_checkpoints, we decided
        # not to slow every other uses case for that.
        # Instead we rely on the fact that when this happens, we actually
        # don't automatically delete files (auto_clear == False).
        assert (
            not self.auto_clear or not self.kept_checkpoints.count(checkpoint_filename)
        ), "Not yet implemented"
        if (
            self.auto_clear
            and len(self.kept_checkpoints) == self.kept_checkpoints.maxlen
        ):
            # We reach the max number of checkpoints we keep around.
            # Delete the oldest one.
            os.remove(self.kept_checkpoints.popleft())
        # Save the new checkpoint.
        self.kept_checkpoints.append(checkpoint_filename)

    def get_last_n(self, num_elements):
        assert 0 < num_elements <= self.kept_checkpoints.maxlen, (
            f"Requested number of elements {num_elements} "
            f"must be between 0 and maxlen {self.kept_checkpoints.maxlen}, "
            f"exclusive"
        )
        # If we ask for more elements than what we currently have, return all
        # of them.
        # Reason why we don't assert unlike for maxlen is because maxlen points
        # out a design issue (the reserved size is too small), whereas the case
        # where we ask more elements than what is currently in the list happens
        # when we print the average of X checkpoints for BLEU, but we haven't
        # yet computed that many checkpoints. We could also assert in this case
        # and fix the caller, but handling it here was just fine!
        start = max(len(self.kept_checkpoints) - num_elements, 0)
        return collections.deque(itertools.islice(self.kept_checkpoints, start, None))


def save_checkpoint(trainer, args, extra_state):
    epoch = extra_state["epoch"]
    batch_offset = extra_state["batch_offset"]
    val_loss = extra_state["val_loss"]

    if args.log_verbose:
        print(
            f"Preparing to save checkpoints for epoch {epoch}, "
            f"offset {batch_offset}. ",
            flush=True
        )

    if "last_checkpoints" not in extra_state:
        if args.generate_bleu_eval_avg_checkpoints < 1:
            raise argparse.ArgumentTypeError(
                "--generate-bleu-eval-avg-checkpoints must be >= 1."
            )
        extra_state["last_checkpoints"] = ManagedCheckpoints(
            max(args.generate_bleu_eval_avg_checkpoints, args.max_checkpoints_kept),
            # Don't auto_clear checkpoints for no_epoch_checkpoints, because
            # we are only going to reuse the same file.
            auto_clear=(
                args.max_checkpoints_kept > 0 and not args.no_epoch_checkpoints
            ),
        )

    # batch_offset being None means that we're at the end of an epoch.
    if batch_offset is None:
        if not args.no_epoch_checkpoints:
            epoch_filename = os.path.join(args.save_dir, f"checkpoint{epoch}.pt")
            save_checkpoint_maybe_continuous(epoch_filename, trainer, extra_state)
            extra_state["last_checkpoints"].append(epoch_filename)

        assert val_loss is not None

        if (
            "checkpoint_lowest_loss" not in extra_state
            or val_loss < extra_state["checkpoint_lowest_loss"]
        ):
            extra_state["checkpoint_lowest_loss"] = val_loss
            best_filename = os.path.join(args.save_dir, "checkpoint_best.pt")
            save_checkpoint_maybe_continuous(best_filename, trainer, extra_state)

    # Otherwise, we're in the middle of an epoch.
    elif not args.no_epoch_checkpoints:
        epoch_filename = os.path.join(
            args.save_dir, f"checkpoint{epoch}_{batch_offset}.pt"
        )
        save_checkpoint_maybe_continuous(epoch_filename, trainer, extra_state)
        extra_state["last_checkpoints"].append(epoch_filename)

    last_filename = os.path.join(args.save_dir, "checkpoint_last.pt")
    save_checkpoint_maybe_continuous(last_filename, trainer, extra_state)

    # This ensures we'll always have at least one checkpoint in the list to use
    # for BLEU eval, even if we're not saving epoch checkpoints.
    if args.no_epoch_checkpoints:
        extra_state["last_checkpoints"].append(epoch_filename)
    if args.log_verbose:
        print(
            f"Finished saving checkpoints for epoch {epoch}, "
            f"offset {batch_offset}.",
            flush=True,
        )


def validate(args, trainer, dataset, subset, extra_state):
    """Evaluate the model on the validation set and return the average loss."""
    epoch = extra_state["epoch"]
    # Initialize dataloader
    max_positions_valid = (
        trainer.get_model().max_encoder_positions(),
        trainer.get_model().max_decoder_positions(),
    )
    itr = dataset.eval_dataloader(
        subset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=max_positions_valid,
        skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test,
        descending=True,  # largest batch first to warm the caching allocator
        shard_id=args.distributed_rank,
        num_shards=args.distributed_world_size,
    )
    progress = progress_bar.build_progress_bar(
        args, itr, epoch, prefix=f"valid on '{subset}' subset", no_progress_bar="simple"
    )

    # reset validation loss meters
    for k in ["valid_loss", "valid_nll_loss"]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    for sample in progress:
        log_output = trainer.valid_step(sample)

        # log mid-validation stats
        stats = get_valid_stats(trainer)
        for k, v in log_output.items():
            if k in ["loss", "nll_loss"]:
                continue
            if "loss" in k:
                extra_meters[k].update(v, log_output["sample_size"])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

    # log validation stats
    stats = get_valid_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    val_loss = stats["valid_loss"]
    val_ppl = stats["valid_ppl"]

    if (
        "validate" not in extra_state
        or val_loss < extra_state["validate"]["lowest_loss"]
    ):
        extra_state["validate"] = {"lowest_loss": val_loss, "num_since_best": 0}
    else:
        extra_state["validate"]["num_since_best"] += 1

    stop_due_to_val_loss = False
    if (
        args.stop_no_best_validate_loss >= 0
        and extra_state["validate"]["num_since_best"] > args.stop_no_best_validate_loss
    ):
        stop_due_to_val_loss = True
        print(
            f"Stopping training due to validation score stagnation - last best "
            f"validation loss of {extra_state['validate']['lowest_loss']} (current loss: {val_loss})"
            f"was {extra_state['validate']['num_since_best']} validations ago."
        )
    return val_loss, val_ppl, stop_due_to_val_loss


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats["valid_loss"] = trainer.get_meter("valid_loss").avg
    if trainer.get_meter("valid_nll_loss").count > 0:
        nll_loss = trainer.get_meter("valid_nll_loss").avg
        stats["valid_nll_loss"] = nll_loss
    else:
        nll_loss = trainer.get_meter("valid_loss").avg
    stats["valid_ppl"] = get_perplexity(nll_loss)
    return stats


def _save_averaged_checkpoint(args, extra_state):
    epoch, offset = extra_state["epoch"], extra_state["batch_offset"]
    if not hasattr(_save_averaged_checkpoint, "last_avg_checkpoints"):
        if args.max_checkpoints_kept == 0:
            raise argparse.ArgumentTypeError("--max-checkpoints-kept must be != 0.")
        _save_averaged_checkpoint.last_avg_checkpoints = ManagedCheckpoints(
            max(args.max_checkpoints_kept, 1), auto_clear=args.max_checkpoints_kept > 0
        )

    last_checkpoints = extra_state["last_checkpoints"].get_last_n(
        1 if args.no_epoch_checkpoints else args.generate_bleu_eval_avg_checkpoints
    )
    if args.log_verbose:
        print(
            f"Reading {len(last_checkpoints)} previous "
            f"checkpoints for averaging in epoch {epoch}, offset {offset}.",
            flush=True,
        )
    averaged_state = average_checkpoints.average_checkpoints(last_checkpoints)
    filename = os.path.join(args.save_dir, f"averaged_checkpoint{epoch}_{offset}.pt")
    _save_averaged_checkpoint.last_avg_checkpoints.append(filename)
    if args.log_verbose:
        print(
            f"Preparing to save averaged checkpoint for "
            f"epoch {epoch}, offset {offset}.",
            flush=True,
        )
    torch.save(averaged_state, filename)
    if args.log_verbose:
        print(
            f"Finished saving averaged checkpoint for "
            f"epoch {epoch}, offset {offset}.",
            flush=True,
        )
    return filename


def calculate_bleu_on_subset(args, dataset, epoch, offset, dataset_split):
    scorer, num_sentences, gen_timer = generate.generate_score(
        args=args, dataset=dataset, dataset_split=dataset_split
    )

    print(
        f"| epoch {epoch:03d} | offset {offset} "
        f"| Eval on {dataset_split} subset "
        f"with beam={args.beam}: {scorer.result_string()}. "
        f"Generated {num_sentences} sentences ({gen_timer.n} tokens) "
        f"in {gen_timer.sum:.1f}s ({1. / gen_timer.avg:.2f} tokens/s).",
        flush=True,
    )
    return scorer.score()


def evaluate_bleu(args, dataset, extra_state):
    epoch, offset = extra_state["epoch"], extra_state["batch_offset"]
    filename = _save_averaged_checkpoint(args, extra_state)
    args.path = [filename]
    val_bleu = calculate_bleu_on_subset(
        args=args,
        dataset=dataset,
        epoch=epoch,
        offset=offset,
        dataset_split=args.valid_subset,
    )

    if (
        "evaluate_bleu" not in extra_state
        or val_bleu > extra_state["evaluate_bleu"]["best"]
    ):
        extra_state["evaluate_bleu"] = {
            "best": val_bleu, "best_epoch": epoch, "num_since_best": 0
        }
        best_filename = os.path.join(args.save_dir, "averaged_checkpoint_best.pt")
        shutil.copy2(filename, best_filename)
    else:
        extra_state["evaluate_bleu"]["num_since_best"] += 1

    stop_due_to_val_bleu = False
    if (
        args.stop_no_best_bleu_eval >= 0
        and extra_state["evaluate_bleu"]["num_since_best"] > args.stop_no_best_bleu_eval
    ):
        stop_due_to_val_bleu = True
        print(
            f"Stopping training due to BLEU score stagnation on valid set - "
            f"last best BLEU score of {extra_state['evaluate_bleu']['best']} "
            f"(current score: {val_bleu}) was"
            f"{extra_state['evaluate_bleu']['num_since_best']} evals ago."
        )
    return val_bleu, stop_due_to_val_bleu


def validate_save_and_evaluate_bleu(
    args,
    trainer,
    dataset,
    extra_state: Dict[str, Any],
    do_validate: bool,
    do_save: bool,
    do_eval_bleu: bool,
) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
    # evaluate on validate set
    val_loss = None
    val_ppl = None
    stop_due_to_val_loss = False
    if do_validate:
        val_loss, val_ppl, stop_due_to_val_loss = validate(
            args=args,
            trainer=trainer,
            dataset=dataset,
            subset=args.valid_subset,
            extra_state=extra_state,
        )
    extra_state["val_loss"] = val_loss

    val_bleu = None
    stop_due_to_val_bleu = False
    if do_save and distributed_utils.is_master(args):
        # save checkpoint
        save_checkpoint(trainer=trainer, args=args, extra_state=extra_state)
        if do_eval_bleu:
            val_bleu, stop_due_to_val_bleu = evaluate_bleu(
                args=args, dataset=dataset, extra_state=extra_state
            )

    return (val_loss, val_ppl, val_bleu, stop_due_to_val_loss or stop_due_to_val_bleu)


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        import signal
        import threading

        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        self.children_pids.append(pid)

    def error_listener(self):
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)


def run(args, error_queue):
    try:
        torch.cuda.set_device(args.device_id)
        args.distributed_rank = distributed_utils.distributed_init(args)
        single_process_main(args)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback

        error_queue.put((args.distributed_rank, traceback.format_exc()))


def main(args):
    # Build vocab from the training corpus. We do this outside of train clones
    # to prevent the clones from having to wait on the master clone building the
    # vocab.
    if args.source_lang is None:
        args.source_lang = "src"
    if args.target_lang is None:
        args.target_lang = "tgt"

    args.source_vocab_file = pytorch_translate_data.build_vocab_if_nonexistent(
        vocab_file=args.source_vocab_file,
        corpus_file=args.train_source_text_file,
        dialect=args.source_lang,
        save_dir=args.save_dir,
        max_vocab_size=args.source_max_vocab_size,
    )
    args.target_vocab_file = pytorch_translate_data.build_vocab_if_nonexistent(
        vocab_file=args.target_vocab_file,
        corpus_file=args.train_target_text_file,
        dialect=args.target_lang,
        save_dir=args.save_dir,
        max_vocab_size=args.target_max_vocab_size,
    )

    # Set distributed training parameters for a single node.
    args.distributed_world_size = torch.cuda.device_count()
    args.distributed_init_method = (f"tcp://localhost:{random.randint(10000, 20000)}")

    if args.distributed_world_size == 1:
        return single_process_main(args)

    mp = multiprocessing.get_context("spawn")

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(args.distributed_world_size):
        args.distributed_rank = i
        args.device_id = i
        procs.append(mp.Process(target=run, args=(args, error_queue), daemon=True))
        procs[i].start()
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


if __name__ == "__main__":
    parser = get_parser_with_args()
    args = parse_args_and_arch(parser)
    main(args)
