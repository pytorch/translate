#!/usr/bin/env python3

import argparse
import builtins as __builtin__
import collections
import datetime
import faulthandler
import math
import multiprocessing.queues as mp_queues
import os
import queue
import random
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from fairseq import (
    criterions,
    data,
    distributed_utils,
    models,
    optim,
    options,
    progress_bar,
    tasks,
    utils,
)
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.trainer import Trainer
from pytorch_translate import char_source_hybrid  # noqa
from pytorch_translate import hybrid_transformer_rnn  # noqa
from pytorch_translate import sequence_criterions  # noqa
from pytorch_translate import transformer  # noqa
from pytorch_translate import weighted_criterions  # noqa
from pytorch_translate import (
    checkpoint,
    constants,
    dictionary as pytorch_translate_dictionary,
    evals,
    multi_model,
    options as pytorch_translate_options,
    preprocess,
)
from pytorch_translate.dual_learning import dual_learning_criterion  # noqa
from pytorch_translate.dual_learning import dual_learning_task  # noqa
from pytorch_translate.research.knowledge_distillation import (  # noqa
    dual_decoder_kd_loss,
    dual_decoder_kd_model,
    hybrid_dual_decoder_kd_model,
    knowledge_distillation_loss,
)
from pytorch_translate.word_prediction import word_prediction_criterion  # noqa
from pytorch_translate.word_prediction import word_prediction_model  # noqa


from pytorch_translate import rnn  # noqa; noqa

# we import semi_supervised here so that the model gets registered in fairseq
# model registry
from pytorch_translate import semi_supervised  # noqa; noqa


from pytorch_translate import char_source_model  # noqa; noqa
from pytorch_translate import char_source_transformer_model  # noqa; noqa


def get_parser_with_args(default_task="pytorch_translate"):
    parser = options.get_parser("Trainer", default_task=default_task)
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


def default_extra_state(args) -> Dict[str, Any]:
    return {
        "epoch": 1,
        "batch_offset": 0,
        "start_time": time.time(),
        "previous_training_time": 0.0,
        "tune_eval": {
            "loss": None,
            "perplexity": None,
            "lowest_loss": None,
            "num_since_best": 0,
        },
        # "last_eval_bleu": 0,
        "tune_bleu": {
            "current": None,
            "best": None,
            "best_epoch": None,
            "num_since_best": 0,
        },
        # The list of checkpoint files is actually managed by the
        # CheckpointManager, which overwrites this placeholder when it saves
        # checkpoints.
        "checkpoint_files": [],
        "training_progress": [],
    }


def update_output(
    args,
    extra_state: Dict[str, Any],
    output_queue: Optional[mp_queues.Queue],
    num_updates: int,
    train_ppl: float,
    wps: Optional[float],
):
    if distributed_utils.is_master(args) and output_queue is not None:
        progress_output: Tuple[int, Dict] = (
            num_updates,
            {
                "train_ppl": train_ppl,
                "tune_ppl": extra_state["tune_eval"]["perplexity"],
                "tune_bleu": extra_state["tune_bleu"]["current"],
                "wps": wps,
                # translation_samples isn't currently used by the queue reader,
                # so just pass None for now until we start needing it.
                "translation_samples": None,
            },
        )
        output_queue.put_nowait(progress_output)
        extra_state["training_progress"].append(progress_output)

    return extra_state


def clear_per_step_extra_state(extra_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clear values in extra_state that are technically only true for a specific
    step (ex: the eval tune loss calculated after 5 train steps is no longer
    accurate after 7 train steps, but might not get updated since we might not
    be doing eval after every step).
    """
    extra_state["tune_eval"]["loss"] = None
    extra_state["tune_eval"]["perplexity"] = None
    extra_state["tune_bleu"]["current"] = None
    return extra_state


def validate_and_set_default_args(args):
    # Prevents generate from printing individual translated sentences when
    # calculating BLEU score.
    args.quiet = True

    pytorch_translate_options.check_unsupported_fairseq_flags(args)

    # Set default init method for multi-GPU training if the user didn't specify
    # them.
    if args.distributed_world_size > 1:
        args.distributed_init_method = (
            f"tcp://localhost:{random.randint(10000, 20000)}"
            if not args.distributed_init_method
            else args.distributed_init_method
        )

        if args.local_num_gpus > args.distributed_world_size:
            raise ValueError(
                f"--local-num-gpus={args.local_num_gpus} must be "
                f"<= --distributed-world-size={args.distributed_world_size}."
            )
        if args.local_num_gpus > torch.cuda.device_count():
            raise ValueError(
                f"--local-num-gpus={args.local_num_gpus} must be "
                f"<= the number of GPUs: {torch.cuda.device_count()}."
            )

    if args.fp16 and getattr(args, "adversary", False):
        print(
            "Warning: disabling fp16 training since it's not supported by AdversarialTrainer."
        )
        args.fp16 = False

    if not args.source_vocab_file:
        args.source_vocab_file = pytorch_translate_dictionary.default_dictionary_path(
            save_dir=args.save_dir, dialect=args.source_lang
        )
    if not args.target_vocab_file:
        args.target_vocab_file = pytorch_translate_dictionary.default_dictionary_path(
            save_dir=args.save_dir, dialect=args.target_lang
        )

    if (
        args.arch == "char_source"
        or args.arch == "char_source_transformer"
        or args.arch == "char_source_hybrid"
    ) and not args.char_source_vocab_file:
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


def setup_training_model(args):
    """Parse args, load dataset, and build model with criterion."""
    if not torch.cuda.is_available():
        print("Warning: training without CUDA is likely to be slow!")
    else:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task and load dataset
    task = tasks.setup_task(args)

    # Build model and criterion
    model = task.build_model(args)
    print("| building criterion")
    criterion = task.build_criterion(args)
    print(f"| model {args.arch}, criterion {criterion.__class__.__name__}")
    print(
        f"| num. model params: \
        {sum(p.numel() for p in model.parameters())}"
    )

    if args.task == constants.SEMI_SUPERVISED_TASK:
        # TODO(T35638969): hide this inside the task itself, just use self.args
        task.load_dataset(
            split=args.train_subset,
            src_bin_path=args.train_source_binary_path,
            tgt_bin_path=args.train_target_binary_path,
            forward_model=task.forward_model,
            backward_model=task.backward_model,
        )
    elif args.task == "pytorch_translate_denoising_autoencoder":
        task.load_dataset(
            split=args.train_subset,
            src_bin_path=args.train_source_binary_path,
            tgt_bin_path=args.train_target_binary_path,
            seed=args.seed,
            use_noiser=True,
        )
    elif args.task == "dual_learning_task":
        task.load_dataset(split=args.train_subset, seed=args.seed)
    elif args.task == "pytorch_translate_knowledge_distillation":
        task.load_dataset(
            split=args.train_subset,
            src_bin_path=args.train_source_binary_path,
            tgt_bin_path=args.train_target_binary_path,
            weights_file=getattr(args, "train_weights_path", None),
            is_train=True,
        )
    else:
        task.load_dataset(
            split=args.train_subset,
            src_bin_path=args.train_source_binary_path,
            tgt_bin_path=args.train_target_binary_path,
            weights_file=getattr(args, "train_weights_path", None),
        )
    if args.task == "dual_learning_task":
        task.load_dataset(split=args.valid_subset, seed=args.seed)
    else:
        task.load_dataset(
            split=args.valid_subset,
            src_bin_path=args.eval_source_binary_path,
            tgt_bin_path=args.eval_target_binary_path,
        )
    return task, model, criterion


def setup_training_state(args, trainer, task, epoch_itr):
    """Set up the directory for saving checkpoints.
    Load pretrained model if specified."""
    os.makedirs(args.save_dir, exist_ok=True)

    # If --restore-file is already present under --save-dir, use that one
    # instead of --pretrained-checkpoint-file. The idea is that
    # --pretrained-checkpoint-file allows the user to specify restoring from a
    # different run's checkpoint (possibly with different training params),
    # while not polluting the previous run's checkpoint directory
    # with new checkpoints. However, if training gets interrupted
    # and the user restarts training, we want to resume from
    # the checkpoints under --save-dir, instead of
    # restarting again from the old run's checkpoint at
    # --pretrained-checkpoint-file.
    #
    # Note that if args.restore_file is an absolute path, os.path.join() will
    # ignore previous directory args and just use the absolute path as is.
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    restore_state = True
    if os.path.isfile(checkpoint_path):
        print(
            f"| Using --save-dir={args.save_dir}, --restore-file={args.restore_file}."
        )
    elif args.pretrained_checkpoint_file and os.path.isfile(
        args.pretrained_checkpoint_file
    ):
        checkpoint_path = args.pretrained_checkpoint_file
        restore_state = args.load_pretrained_checkpoint_state
        print(
            f"| Using --pretrained-checkpoint-file={args.pretrained_checkpoint_file}, "
            f"--load-pretrained-checkpoint-state={args.load_pretrained_checkpoint_state}."
        )

    extra_state = default_extra_state(args)
    if not os.path.isfile(checkpoint_path) and args.multi_model_restore_files:
        print(f"| Restoring individual models from {args.multi_model_restore_files}")
        multi_model.import_individual_models(args.multi_model_restore_files, trainer)
    else:
        loaded, loaded_extra_state = checkpoint.load_existing_checkpoint(
            checkpoint_path=checkpoint_path,
            trainer=trainer,
            restore_state=restore_state,
        )
        if loaded_extra_state:
            extra_state.update(loaded_extra_state)

    # Reset the start time for the current training run.
    extra_state["start_time"] = time.time()
    # Removes all uses of the deprecated ManagedCheckpoints
    if "last_checkpoints" in extra_state:
        del extra_state["last_checkpoints"]

    # Skips printing all training progress to prevent log spam.
    training_progress = extra_state["training_progress"]
    extra_state["training_progress"] = (
        ["...truncated...", training_progress[-1]] if len(training_progress) > 0 else []
    )
    print(f"| extra_state: {extra_state}")
    extra_state["training_progress"] = training_progress

    epoch = extra_state["epoch"]
    if extra_state["batch_offset"] == 0:
        epoch -= 1  # this will be incremented when we call epoch_itr.next_epoch_itr()
    epoch_itr.load_state_dict(
        {"epoch": epoch, "iterations_in_epoch": extra_state["batch_offset"]}
    )

    checkpoint_manager = None
    if distributed_utils.is_master(args):
        checkpoint_manager = checkpoint.CheckpointManager(
            num_avg_checkpoints=args.num_avg_checkpoints,
            auto_clear_checkpoints=args.auto_clear_checkpoints,
            log_verbose=args.log_verbose,
            checkpoint_files=extra_state["checkpoint_files"],
        )

    return extra_state, epoch_itr, checkpoint_manager


def build_trainer(args, task, model, criterion, trainer_class):
    """ Build trainer with provided trainer_class, and set up training state.
    """
    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(), model.max_positions()
    )
    dummy_batch = task.dataset("train").get_dummy_batch(args.max_tokens, max_positions)
    oom_batch = task.dataset("train").get_dummy_batch(1, max_positions)

    # Build trainer
    trainer = trainer_class(args, task, model, criterion, dummy_batch, oom_batch)

    print(
        f"| training on {args.distributed_world_size} total GPUs "
        f"({torch.cuda.device_count()} GPUs locally on this machine).\n"
        f"| max tokens per GPU = {args.max_tokens} and \
        max sentences per GPU = {args.max_sentences}",
        flush=True,
    )

    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    )
    return trainer, epoch_itr


def setup_training(args, trainer_class=None):
    """ Perform several steps:
    - build model using provided criterion and task
    - load data
    - build trainer
    """

    # Overrides the default print() to always prepend the timestamp for more
    # informative logging.
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        if "file" not in kwargs:
            builtin_print(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}]",
                *args,
                **kwargs,
            )
        else:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    task, model, criterion = setup_training_model(args)
    if trainer_class is None:
        trainer_class = Trainer

    trainer, epoch_itr = build_trainer(
        args=args,
        task=task,
        model=model,
        criterion=criterion,
        trainer_class=trainer_class,
    )

    return trainer, task, epoch_itr


def create_prune_masks(args, trainer):
    """Generates binary masks for setting model weights to zero"""
    assert (
        args.pruning_percentile > 0 and args.pruning_percentile < 100
    ), "--pruning-percentile must be in (0, 100)"
    all_params = []
    if args.parameters_to_prune == "all":
        parameter_name = "weight"
    elif args.parameters_to_prune == "embed":
        parameter_name = "embed_tokens"
    elif args.parameters_to_prune == "lstm":
        parameter_name = "weight_"
    for name, params in trainer.model.named_parameters():
        if parameter_name in name:
            all_params.append(np.abs(np.reshape(params.data, (-1, 1))))
    threshold = np.percentile(np.vstack(all_params), args.pruning_percentile)

    prune_masks = {}
    for name, params in trainer.model.named_parameters():
        if parameter_name in name:
            prune_masks[name] = np.abs(params.data) < threshold

    return prune_masks


def apply_prune_masks(prune_masks, trainer):
    """Selectively sets model weights to zero using a binary mask."""

    for name, params in trainer.model.named_parameters():
        if name in prune_masks:
            params.data[prune_masks[name]] = 0.0


def train(
    args,
    extra_state: Dict[str, Any],
    trainer,
    task,
    epoch_itr,
    checkpoint_manager: Optional[checkpoint.CheckpointManager],
    output_queue: Optional[mp_queues.Queue] = None,
    **train_step_kwargs,
):
    # offset for current epoch (may be different from checkpoint offset)
    starting_offset = extra_state["batch_offset"]

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    stop_training_mid_epoch = False
    stop_training_end_of_epoch = False

    do_prune = args.pruning_percentile > 0
    if do_prune:
        prune_masks = create_prune_masks(args, trainer)
        apply_prune_masks(prune_masks, trainer)

    while lr > args.min_lr and extra_state["epoch"] <= max_epoch:
        """Train the model for one epoch."""

        itr, progress, extra_meters = setup_epoch(
            args=args, epoch_itr=epoch_itr, trainer=trainer
        )

        for i, samples in enumerate(progress, start=starting_offset):
            clear_per_step_extra_state(extra_state)
            extra_state["num_iterations"] = extra_state.get("num_iterations", 0) + 1
            if (
                train_step_kwargs is not None
                and "augment_adv" in train_step_kwargs.keys()
            ):
                train_step_kwargs["augment_adv"] = (
                    extra_state["num_iterations"] > args.warmup_steps
                )
            try:
                log_output = trainer.train_step(samples, **train_step_kwargs)
            # Fairseq's fp16_trainer raises this uncommon error to indicate
            # that we should stop training.
            except FloatingPointError as e:
                print(f"Stopping training due to: {e}.")
                stop_training_mid_epoch = True
                break

            if do_prune:
                apply_prune_masks(prune_masks, trainer)

            if i == starting_offset:
                # ignore the first mini-batch in words-per-second calculation
                trainer.get_meter("wps").reset()

            # Clear any remaining metrics from previous steps. This should already
            # have been done before, but just in case - to make sure we catch
            # any case where extra_case does not get populated correctly.
            extra_state = clear_per_step_extra_state(extra_state)
            extra_state["batch_offset"] = i + 1
            (
                extra_state,
                stop_training_mid_epoch,
                translation_samples,
            ) = evals.save_and_eval(
                args=args,
                trainer=trainer,
                task=task,
                extra_state=extra_state,
                checkpoint_manager=checkpoint_manager,
            )

            # This should come after save_and_eval. Even if log_output is None,
            # meaning that there was an overflow,  We should still run
            # save_and_eval to sync all_reduce and then skip the batch.
            if log_output is None:
                # This indicates that the batch was skipped, typically
                # because of OOM or FP16 overflow.
                continue

            train_stats = evals.log_mid_epoch_stats(
                trainer=trainer,
                progress=progress,
                extra_meters=extra_meters,
                log_output=log_output,
            )
            extra_state = update_output(
                args=args,
                extra_state=extra_state,
                output_queue=output_queue,
                num_updates=trainer.get_num_updates(),
                train_ppl=train_stats["ppl"],
                # We only report wps at the end of an epoch, since
                # the meter gets reset at the start of every epoch.
                wps=None,
            )

            if (
                args.save_interval_updates > 0
                and extra_state["num_iterations"] % args.save_interval_updates == 0
                and args.shrink_lr_no_best_bleu_eval > 0
                and extra_state["tune_bleu"]["num_since_best"]
                > args.shrink_lr_no_best_bleu_eval
            ):
                current_lr = trainer.optimizer.get_lr()
                trainer.optimizer.set_lr(current_lr * args.lr_shrink)
                lr = trainer.optimizer.get_lr()
                print(f"Decayed lr from {current_lr} to {lr}.")

            if stop_training_mid_epoch:
                break

        # log end-of-epoch stats
        train_stats = evals.log_end_epoch_stats(
            trainer=trainer, progress=progress, extra_meters=extra_meters
        )

        # batch_offset being None denotes the end of an epoch.
        extra_state["batch_offset"] = None
        (
            extra_state,
            stop_training_end_of_epoch,
            translation_samples,
        ) = evals.save_and_eval(
            args=args,
            trainer=trainer,
            task=task,
            extra_state=extra_state,
            end_of_epoch=True,
            checkpoint_manager=checkpoint_manager,
        )
        extra_state = update_output(
            args=args,
            extra_state=extra_state,
            output_queue=output_queue,
            num_updates=trainer.get_num_updates(),
            train_ppl=train_stats["ppl"],
            wps=train_stats["wps"],
        )

        if stop_training_mid_epoch or stop_training_end_of_epoch:
            break

        lr = trainer.lr_step(extra_state["epoch"], extra_state["tune_eval"]["loss"])
        extra_state["epoch"] += 1
        extra_state["batch_offset"] = 0
        starting_offset = 0

    train_meter.stop()
    print(f"| done training in {train_meter.sum:.1f} seconds")
    print(
        f"| Best BLEU score of {extra_state['tune_bleu']['best']} was from "
        f"epoch {extra_state['tune_bleu']['best_epoch']}"
    )


def setup_epoch(args, epoch_itr, trainer):
    """Sets up data and progress meters for one epoch."""
    # Update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    # Initialize dataloader, starting at batch_offset
    itr = epoch_itr.next_epoch_itr()
    itr = data.iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar="simple"
    )

    # reset training meters
    for k in [
        "train_loss",
        "train_nll_loss",
        "wps",
        "ups",
        "wpb",
        "bsz",
        "gnorm",
        "clip",
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    return itr, progress, extra_meters


def single_process_main(args, trainer_class=Trainer, **train_step_kwargs):
    """Train the model for multiple epochs."""
    pytorch_translate_options.print_args(args)
    trainer, task, epoch_itr = setup_training(args, trainer_class)
    extra_state, epoch_itr, checkpoint_manager = setup_training_state(
        args=args, trainer=trainer, task=task, epoch_itr=epoch_itr
    )
    train(
        args=args,
        extra_state=extra_state,
        trainer=trainer,
        task=task,
        epoch_itr=epoch_itr,
        checkpoint_manager=checkpoint_manager,
        **train_step_kwargs,
    )


def multi_process_train(
    device_id: int,
    args,
    output_queue: Optional[mp_queues.Queue],
    start_rank: int = 0,
    init_fn: Optional[Callable[[], None]] = None,
    trainer_class=None,
    train_step_kwargs=None,
):
    # Enable faulthandler for better Python tracebacks upon segfaults under
    # multiprocessing. Without this, the stack trace only shows the
    # SpawnContext.join() call, rather than the actual line where the child
    # process segfaulted.
    faulthandler.enable(all_threads=True)

    if init_fn:
        init_fn()
    args.device_id = device_id
    args.distributed_rank = start_rank + device_id

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)

    trainer, task, epoch_itr = setup_training(args, trainer_class)
    # Distributed_init does initialization and works as a barrier.
    # Therefore, any expensive data preprocessing should happen before.
    if args.distributed_world_size > 1:
        args.distributed_rank = distributed_utils.distributed_init(args)
    extra_state, epoch_itr, checkpoint_manager = setup_training_state(
        args=args, trainer=trainer, task=task, epoch_itr=epoch_itr
    )

    # Replay previous training progress so the output_queue contains all
    # previous training progress even when we resume training from an existing
    # checkpoint.
    if distributed_utils.is_master(args) and output_queue is not None:
        for progress_output in extra_state["training_progress"]:
            output_queue.put_nowait(progress_output)

    train(
        args=args,
        extra_state=extra_state,
        trainer=trainer,
        task=task,
        epoch_itr=epoch_itr,
        checkpoint_manager=checkpoint_manager,
        output_queue=output_queue,
        **train_step_kwargs,
    )


def multi_process_main(
    args: Any,
    start_rank: int = 0,
    init_fn: Optional[Callable[[], None]] = None,
    trainer_class=None,
    **train_step_kwargs,
):
    pytorch_translate_options.print_args(args)
    output_queue = torch.multiprocessing.get_context("spawn").Queue()
    # Train with multiprocessing.
    spawn_context = torch.multiprocessing.spawn(
        fn=multi_process_train,
        args=(
            args,
            output_queue,
            start_rank,
            init_fn,
            trainer_class,
            train_step_kwargs,
        ),
        nprocs=args.local_num_gpus,
        # We don't block here to allow caller to process output_queue in
        # parallel with training.
        join=False,
    )
    return (spawn_context, output_queue)


def main(args, trainer_class=Trainer, **train_step_kwargs):
    # We preprocess the data (generating vocab files and binarized data files
    # if needed) outside of the train processes to prevent them from having to
    # wait while the master process is doing this.
    preprocess.preprocess_corpora(args)

    if args.distributed_world_size == 1:
        single_process_main(args, trainer_class, **train_step_kwargs)
    else:
        spawn_context, output_queue = multi_process_main(args=args, start_rank=0)

        while not spawn_context.join(timeout=30):
            # Periodically clears the output queue to ensure that the processes
            # don't deadlock due to queue buffer being full. This is also
            # necessary to ensure that processes join correctly, since a process
            # may not terminate until all items it put on the queue have been
            # consumed (per
            # https://docs.python.org/3/library/multiprocessing.html#all-start-methods).
            try:
                while True:
                    output_queue.get_nowait()
            except queue.Empty:
                pass


if __name__ == "__main__":
    parser = get_parser_with_args()
    args = options.parse_args_and_arch(parser)
    validate_and_set_default_args(args)
    main(args)
