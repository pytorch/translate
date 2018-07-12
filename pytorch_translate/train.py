#!/usr/bin/env python3

import argparse
import collections
import itertools
import math
import multiprocessing
import numpy as np
import os
import random
import shutil
import signal
import tempfile
import time
import torch

from typing import Any, Dict, Optional, Tuple

from fairseq import (
    criterions,
    distributed_utils,
    models,
    optim,
    options,
    progress_bar,
    utils,
)
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.trainer import Trainer

from pytorch_translate import average_checkpoints
from pytorch_translate import char_source_model  # noqa
from pytorch_translate import constants
from pytorch_translate import data as pytorch_translate_data
from pytorch_translate import options as pytorch_translate_options
from pytorch_translate import dictionary as pytorch_translate_dictionary
from pytorch_translate import generate
from pytorch_translate import preprocess
from pytorch_translate import rnn  # noqa
from pytorch_translate import weighted_criterions  # noqa
from pytorch_translate.utils import ManagedCheckpoints  # noqa
from pytorch_translate import multi_model
from pytorch_translate.research.word_prediction import word_prediction_criterion  # noqa
from pytorch_translate.research.word_prediction import word_prediction_model  # noqa
from pytorch_translate.research.knowledge_distillation import (  # noqa
    knowledge_distillation_loss
)


def get_parser_with_args():
    parser = options.get_parser("Trainer")
    pytorch_translate_options.add_verbosity_args(parser, train=True)
    pytorch_translate_options.add_dataset_args(parser, train=True, gen=True)
    options.add_distributed_training_args(parser)
    # Adds args related to training (validation and stopping criterions).
    optimization_group = options.add_optimization_args(parser)
    pytorch_translate_options.expand_optimization_args(optimization_group)
    # Adds args related to checkpointing.
    checkpointing_group = options.add_checkpoint_args(parser)
    pytorch_translate_options.expand_checkpointing_args(checkpointing_group)
    # Add model related args
    options.add_model_args(parser)
    # Adds args for generating intermediate BLEU eval while training.
    generation_group = options.add_generation_args(parser)
    pytorch_translate_options.expand_generation_args(generation_group, train=True)
    # Adds args related to input data files (preprocessing, numberizing, and
    # binarizing text files; creating vocab files)
    pytorch_translate_options.add_preprocessing_args(parser)
    return parser


def load_existing_checkpoint(checkpoint_path, trainer, restore_state=True):
    extra_state = None
    loaded = False
    if restore_state:
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is None:
            loaded = False
            print(f"Failed to load checkpoint and state from {checkpoint_path}.")
        else:
            loaded = True
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
        # TODO(weiho): use trainer.load_checkpoint(load_optim=False) after
        # that's been synced to open-source fairseq.
        dummy_state, _, _ = utils.load_model_state(
            checkpoint_path, trainer.model, cuda_device=torch.cuda.current_device()
        )
        trainer.optimizer = optim.build_optimizer(
            trainer.args, trainer.model.parameters()
        )
        trainer.lr_scheduler = optim.lr_scheduler.build_lr_scheduler(
            trainer.args, trainer.optimizer
        )
        trainer._optim_history = []

        if dummy_state is None:
            loaded = False
            print(f"Failed to load checkpoint weights from {checkpoint_path}.")
        else:
            loaded = True
            print(f"Loaded checkpoint weights from {checkpoint_path}.")

    if extra_state is None:
        extra_state = {
            "epoch": 1,
            "batch_offset": 0,
            "val_loss": None,
            "start_time": time.time(),
            "last_bleu_eval": 0,
        }

    return loaded, extra_state


def validate_and_set_default_args(args):
    # Prevents generate from printing individual translated sentences when
    # calculating BLEU score.
    args.quiet = True

    if not args.source_vocab_file:
        args.source_vocab_file = pytorch_translate_dictionary.default_dictionary_path(
            save_dir=args.save_dir, dialect=args.source_lang
        )
    if not args.target_vocab_file:
        args.target_vocab_file = pytorch_translate_dictionary.default_dictionary_path(
            save_dir=args.save_dir, dialect=args.target_lang
        )

    if args.arch == "char_source" and not args.char_source_vocab_file:
        args.char_source_vocab_file = pytorch_translate_dictionary.default_char_dictionary_path(
            save_dir=args.save_dir, dialect=args.source_lang
        )

    pytorch_translate_options.validate_preprocessing_args(args)
    pytorch_translate_options.validate_generation_args(args)
    if args.multiling_encoder_lang and not args.multiling_source_vocab_file:
        args.multiling_source_vocab_file = [
            pytorch_translate_dictionary.default_dictionary_path(
                save_dir=args.save_dir, dialect=f"src-{l}"
            )
            for l in args.multiling_encoder_lang
        ]
    if args.multiling_decoder_lang and not args.multiling_target_vocab_file:
        args.multiling_target_vocab_file = [
            pytorch_translate_dictionary.default_dictionary_path(
                save_dir=args.save_dir, dialect=f"trg-{l}"
            )
            for l in args.multiling_decoder_lang
        ]


def setup_training(args):
    """Parse args, load dataset, and load model trainer."""
    if not torch.cuda.is_available():
        raise NotImplementedError("Training on CPU is not supported")
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Load dataset
    splits = [args.train_subset, args.valid_subset]

    validate_and_set_default_args(args)

    train_corpus = pytorch_translate_data.ParallelCorpusConfig(
        source=pytorch_translate_data.CorpusConfig(
            dialect=args.source_lang, data_file=args.train_source_binary_path
        ),
        target=pytorch_translate_data.CorpusConfig(
            dialect=args.target_lang, data_file=args.train_target_binary_path
        ),
        weights_file=args.train_weights_path
        if hasattr(args, "train_weights_path")
        else None,
    )

    eval_corpus = pytorch_translate_data.ParallelCorpusConfig(
        source=pytorch_translate_data.CorpusConfig(
            dialect=args.source_lang, data_file=args.eval_source_binary_path
        ),
        target=pytorch_translate_data.CorpusConfig(
            dialect=args.target_lang, data_file=args.eval_target_binary_path
        ),
        weights_file=None,
    )

    if args.log_verbose:
        print("Starting to load binarized data files.", flush=True)
    use_char_source = args.arch == "char_source"
    dataset = pytorch_translate_data.load_binarized_dataset(
        train_corpus=train_corpus,
        eval_corpus=eval_corpus,
        train_split=args.train_subset,
        eval_split=args.valid_subset,
        args=args,
        use_char_source=use_char_source,
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
    print("building criterion")
    criterion = criterions.build_criterion(args, dataset.src_dict, dataset.dst_dict)
    print(f"| model {args.arch}, criterion {criterion.__class__.__name__}")
    print(
        f"| num. model params: \
        {sum(p.numel() for p in model.parameters())}"
    )

    # Build trainer
    trainer = Trainer(args, model, criterion)
    print(f"| training on {args.distributed_world_size} GPUs")
    print(
        f"| max tokens per GPU = {args.max_tokens} and \
        max sentences per GPU = {args.max_sentences}",
        flush=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # If --restore-file is already present under --save-dir, use that one
    # instead of the --restore-file that may be present under
    # --restore-checkpoint-dir. The idea is that --restore-checkpoint-dir
    # allows the user to specify restoring from a different run's
    # checkpoint (possibly with different training params), while not
    # polluting the previous run's checkpoint directory with new checkpoints.
    # However, if training gets interrupted and the user restarts training,
    # we want to resume from the checkpoints under --save-dir, instead of
    # restarting again from the old run's checkpoint under
    # --restore-checkpoint-dir.
    #
    # Note that if args.restore_file is an absolute path, os.path.join() will
    # ignore previous directory args and just use the absolute path as is.
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.exists(checkpoint_path):
        print(f"Using --save-dir={args.save_dir}, --restore-file={args.restore_file}.")
    elif args.restore_checkpoint_dir:
        checkpoint_path = os.path.join(args.restore_checkpoint_dir, args.restore_file)
        print(
            f"Using --restore-checkpoint-dir={args.restore_checkpoint_dir}, "
            f"--restore-file={args.restore_file}."
        )

    if not os.path.isfile(checkpoint_path) and args.multi_model_restore_files:
        print(f"| Restoring individual models from {args.multi_model_restore_files}")
        extra_state = multi_model.import_individual_models(
            args.multi_model_restore_files, trainer
        )
    else:
        loaded, extra_state = load_existing_checkpoint(
            checkpoint_path=checkpoint_path,
            trainer=trainer,
            restore_state=args.restore_checkpoint_state,
        )
        if loaded:
            args.path = [checkpoint_path]
            calculate_bleu_on_subset(
                args=args,
                dataset=dataset,
                epoch_str="initial loaded checkpoint",
                offset=None,
                dataset_split=args.valid_subset,
            )

    return extra_state, trainer, dataset


def prune(args, trainer):
    """Sets some model weights to zero based on pruning scheme"""
    assert (
        args.pruning_percentile > 0 and args.pruning_percentile < 100
    ), "--pruning-percentile must be in (0, 100)"
    all_params = []
    for name, params in trainer.model.named_parameters():
        if "weight" in name:
            all_params.append(np.abs(np.reshape(params.data, (-1, 1))))
    threshold = np.percentile(np.vstack(all_params), args.pruning_percentile)

    prune_masks = {}
    for name, params in trainer.model.named_parameters():
        if "weight" in name:
            prune_masks[name] = np.abs(params.data) < threshold
            params.data[prune_masks[name]] = 0.0

    return prune_masks


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
    do_prune = args.pruning_percentile > 0
    extra_state["retraining"] = False
    prune_masks = None
    stop_training_mid_epoch = False
    stop_training_end_of_epoch = False
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

            if extra_state["retraining"]:
                for name, params in trainer.model.named_parameters():
                    if "weight" in name:
                        params.data[prune_masks[name]] = 0.0

            log_output = trainer.train_step(sample)

            train_stats = log_mid_epoch_stats(
                trainer=trainer,
                progress=progress,
                extra_meters=extra_meters,
                log_output=log_output,
            )

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
                _,
                val_ppl,
                val_bleu,
                stop_training_mid_epoch,
                translation_samples,
                lr,
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
                    "translation_samples": translation_samples,
                },
            )

            stop_training_mid_epoch = (
                stop_training_mid_epoch
                or is_training_over_time_limit(
                    extra_state["start_time"], args.stop_time_hr
                )
            )
            if stop_training_mid_epoch:
                break

        # log end-of-epoch stats
        train_stats = log_end_epoch_stats(
            trainer=trainer, progress=progress, extra_meters=extra_meters
        )

        # Run a training step if not stopping mid-epoch.
        if not stop_training_mid_epoch:
            # batch_offset being None denotes the end of an epoch.
            extra_state["batch_offset"] = None
            (
                val_loss,
                val_ppl,
                val_bleu,
                stop_training_end_of_epoch,
                translation_samples,
                lr,
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
                    "translation_samples": translation_samples,
                },
            )
        if stop_training_mid_epoch or stop_training_end_of_epoch:
            if do_prune and not extra_state["retraining"]:
                lr *= args.retrain_lr_ratio
                extra_state["validate"]["lowest_loss"] = np.inf
                extra_state["evaluate_bleu"]["best"] = 0
                stop_training_mid_epoch = False
                stop_training_end_of_epoch = False
                prune_masks = prune(args, trainer)
                extra_state["retraining"] = True
                print("| Finished pruning and switching to retraining")
            else:
                break

        lr = trainer.lr_step(extra_state["epoch"], val_loss)
        extra_state["epoch"] += 1
        extra_state["batch_offset"] = 0
        starting_offset = 0

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


def save_checkpoint(trainer, args, extra_state):
    epoch = extra_state["epoch"]
    batch_offset = extra_state["batch_offset"]
    val_loss = extra_state["val_loss"]

    if args.log_verbose:
        print(
            f"Preparing to save checkpoints for epoch {epoch}, "
            f"offset {batch_offset}. ",
            flush=True,
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
            trainer.save_checkpoint(epoch_filename, extra_state)
            extra_state["last_checkpoints"].append(epoch_filename)

        assert val_loss is not None

        if (
            "checkpoint_lowest_loss" not in extra_state
            or val_loss < extra_state["checkpoint_lowest_loss"]
        ):
            extra_state["checkpoint_lowest_loss"] = val_loss
            best_filename = os.path.join(args.save_dir, "checkpoint_best.pt")
            trainer.save_checkpoint(best_filename, extra_state)

    # Otherwise, we're in the middle of an epoch.
    elif not args.no_epoch_checkpoints:
        epoch_filename = os.path.join(
            args.save_dir, f"checkpoint{epoch}_{batch_offset}.pt"
        )
        trainer.save_checkpoint(epoch_filename, extra_state)
        extra_state["last_checkpoints"].append(epoch_filename)

    last_filename = os.path.join(args.save_dir, "checkpoint_last.pt")
    trainer.save_checkpoint(last_filename, extra_state)

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
            f"validation loss of {extra_state['validate']['lowest_loss']} (current loss: {val_loss}) "
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
    utils.torch_persistent_save(averaged_state, filename)
    if args.log_verbose:
        print(
            f"Finished saving averaged checkpoint for "
            f"epoch {epoch}, offset {offset}.",
            flush=True,
        )
    return filename


def calculate_bleu_on_subset(args, dataset, epoch_str: str, offset, dataset_split):
    scorer, num_sentences, gen_timer, translation_samples = generate.generate_score(
        args=args, dataset=dataset, dataset_split=dataset_split
    )

    print(
        f"| epoch {epoch_str} | offset {offset} "
        f"| Eval on {dataset_split} subset "
        f"with beam={args.beam}: {scorer.result_string()}. "
        f"Generated {num_sentences} sentences ({gen_timer.n} tokens) "
        f"in {gen_timer.sum:.1f}s ({1. / gen_timer.avg:.2f} tokens/s).",
        flush=True,
    )
    return scorer.score(), translation_samples


def evaluate_bleu(args, dataset, extra_state):
    epoch, offset = extra_state["epoch"], extra_state["batch_offset"]
    filename = _save_averaged_checkpoint(args, extra_state)
    args.path = [filename]
    val_bleu, translation_samples = calculate_bleu_on_subset(
        args=args,
        dataset=dataset,
        epoch_str=f"{epoch:03d}",
        offset=offset,
        dataset_split=args.valid_subset,
    )

    if (
        "evaluate_bleu" not in extra_state
        or val_bleu > extra_state["evaluate_bleu"]["best"]
    ):
        extra_state["evaluate_bleu"] = {
            "best": val_bleu,
            "best_epoch": epoch,
            "num_since_best": 0,
        }
        best_filename = os.path.join(
            args.save_dir, constants.AVERAGED_CHECKPOINT_BEST_FILENAME
        )
        shutil.copy2(filename, best_filename)
    else:
        extra_state["evaluate_bleu"]["num_since_best"] += 1

    decay_lr = False
    if (
        args.shrink_lr_no_best_bleu_eval > 0
        and extra_state["evaluate_bleu"]["num_since_best"]
        > args.shrink_lr_no_best_bleu_eval
    ):
        decay_lr = True

    stop_due_to_val_bleu = False
    if (
        args.stop_no_best_bleu_eval >= 0
        and extra_state["evaluate_bleu"]["num_since_best"] > args.stop_no_best_bleu_eval
    ):
        stop_due_to_val_bleu = True
        print(
            f"Stopping training due to BLEU score stagnation on valid set - "
            f"last best BLEU score of {extra_state['evaluate_bleu']['best']} "
            f"(current score: {val_bleu}) was "
            f"{extra_state['evaluate_bleu']['num_since_best']} evals ago."
        )
    return val_bleu, stop_due_to_val_bleu, translation_samples, decay_lr


def validate_save_and_evaluate_bleu(
    args,
    trainer,
    dataset,
    extra_state: Dict[str, Any],
    do_validate: bool,
    do_save: bool,
    do_eval_bleu: bool,
) -> Tuple[
    Optional[float], Optional[float], Optional[float], bool, Optional[list], float
]:
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

    lr = trainer.optimizer.get_lr()
    val_bleu = None
    stop_due_to_val_bleu = False
    translation_samples = None
    if do_save and distributed_utils.is_master(args):
        # save checkpoint
        save_checkpoint(trainer=trainer, args=args, extra_state=extra_state)
        if do_eval_bleu:
            (
                val_bleu,
                stop_due_to_val_bleu,
                translation_samples,
                decay_lr,
            ) = evaluate_bleu(args=args, dataset=dataset, extra_state=extra_state)
            if decay_lr:
                current_lr = lr
                trainer.optimizer.set_lr(lr * args.lr_shrink)
                lr = trainer.optimizer.get_lr()
                print(f"Decay lr from {current_lr} to {lr}.")

    return (
        val_loss,
        val_ppl,
        val_bleu,
        stop_due_to_val_loss or stop_due_to_val_bleu,
        translation_samples,
        lr,
    )


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


def run(args, single_process_train, error_queue):
    try:
        torch.cuda.set_device(args.device_id)
        args.distributed_rank = distributed_utils.distributed_init(args)
        single_process_train(args)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback

        error_queue.put((args.distributed_rank, traceback.format_exc()))


def main(args, single_process_train):
    # We preprocess the data (generating vocab files and binarized data files
    # if needed) outside of the train clones to prevent them from having to
    # wait while the master clone is doing this.
    preprocess.preprocess_corpora(args)

    # Set distributed training parameters for a single node.
    args.distributed_world_size = torch.cuda.device_count()
    args.distributed_init_method = f"tcp://localhost:{random.randint(10000, 20000)}"

    if args.distributed_world_size == 1:
        return single_process_train(args)

    mp = multiprocessing.get_context("spawn")

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(args.distributed_world_size):
        args.distributed_rank = i
        args.device_id = i
        procs.append(
            mp.Process(
                target=run, args=(args, single_process_train, error_queue), daemon=True
            )
        )
        procs[i].start()
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


if __name__ == "__main__":
    parser = get_parser_with_args()
    args = options.parse_args_and_arch(parser)
    validate_and_set_default_args(args)
    pytorch_translate_options.print_args(args)
    main(args, single_process_main)
