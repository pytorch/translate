#!/usr/bin/env python3

import math
import time
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from fairseq import distributed_utils, progress_bar, utils
from fairseq.meters import AverageMeter
from pytorch_translate import checkpoint, generate, utils as pytorch_translate_utils
from pytorch_translate.dual_learning.dual_learning_task import DualLearningTask
from pytorch_translate.tasks.pytorch_translate_multi_task import (
    PyTorchTranslateMultiTask,
)


def log_mid_epoch_stats(trainer, progress, extra_meters, log_output):
    stats = get_training_stats(trainer)
    for k, v in log_output.items():
        if k in ["loss", "nll_loss", "ntokens", "nsentences", "sample_size"]:
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
    stats = OrderedDict()
    if trainer.get_meter("train_loss") is not None:
        avg = trainer.get_meter("train_loss").avg
        if avg is not None:
            stats["loss"] = f"{avg:.3f}"
    if trainer.get_meter("train_nll_loss").count > 0:
        nll_loss = trainer.get_meter("train_nll_loss").avg
        stats["nll_loss"] = f"{nll_loss:.3f}"
    else:
        nll_loss = trainer.get_meter("train_nll_loss").avg
    stats["ppl"] = get_perplexity(nll_loss) if nll_loss is not None else -1.0
    if trainer.get_meter("wps") is not None:
        stats["wps"] = (
            round(utils.item(trainer.get_meter("wps").avg))
            if trainer.get_meter("wps").avg
            else None
        )
    if trainer.get_meter("ups") is not None:
        stats["ups"] = (
            f"{trainer.get_meter('ups').avg:.1f}"
            if trainer.get_meter("ups").avg
            else None
        )
    if trainer.get_meter("wpb") is not None:
        stats["wpb"] = (
            round(utils.item(trainer.get_meter("wpb").avg))
            if trainer.get_meter("wpb").avg
            else None
        )
    if trainer.get_meter("bsz") is not None:
        stats["bsz"] = (
            round(utils.item(trainer.get_meter("bsz").avg))
            if trainer.get_meter("bsz").avg
            else None
        )
    stats["num_updates"] = trainer.get_num_updates()
    stats["lr"] = trainer.get_lr()
    if trainer.get_meter("gnorm") is not None:
        stats["gnorm"] = (
            f"{trainer.get_meter('gnorm').avg:.3f}"
            if trainer.get_meter("gnorm").avg
            else None
        )
    if trainer.get_meter("clip") is not None:
        stats["clip"] = (
            f"{trainer.get_meter('clip').avg:.0%}"
            if trainer.get_meter("clip").avg
            else None
        )
    if trainer.get_meter("oom") is not None:
        stats["oom"] = (
            trainer.get_meter("oom").avg if trainer.get_meter("oom").avg else None
        )
    if trainer.get_meter("loss_scale") is not None:
        stats["loss_scale"] = (
            f"{trainer.get_meter('loss_scale').avg:.3f}"
            if trainer.get_meter("loss_scale").avg
            else None
        )
    if trainer.get_meter("wall") is not None:
        stats["wall"] = (
            round(utils.item(trainer.get_meter("wall").elapsed_time))
            if trainer.get_meter("wall").elapsed_time
            else None
        )
    if trainer.get_meter("train_wall") is not None:
        stats["train_wall"] = (
            round(utils.item(trainer.get_meter("train_wall").sum))
            if trainer.get_meter("train_wall").sum
            else None
        )
    return stats


def get_valid_stats(trainer):
    stats = OrderedDict()
    stats["valid_loss"] = trainer.get_meter("valid_loss").avg
    if trainer.get_meter("valid_nll_loss").count > 0:
        nll_loss = trainer.get_meter("valid_nll_loss").avg
        stats["valid_nll_loss"] = nll_loss
    else:
        nll_loss = trainer.get_meter("valid_loss").avg
    stats["valid_ppl"] = get_perplexity(nll_loss)
    return stats


def get_perplexity(loss):
    try:
        return f"{math.pow(2, loss):.2f}"
    except OverflowError:
        return float("inf")


def eval_tune_loss(args, trainer, task, subset, extra_state):
    """Evaluate the model on the validation set and return the average loss."""
    # Initialize dataloader
    itr = task.get_batch_iterator(
        dataset=task.dataset(subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), trainer.get_model().max_positions()
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.build_progress_bar(
        args=args,
        iterator=itr,
        epoch=extra_state["epoch"],
        prefix=f"valid on '{subset}' subset",
        no_progress_bar="simple",
    )

    # reset validation loss meters
    for k in ["valid_loss", "valid_nll_loss"]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = defaultdict(lambda: AverageMeter())
    for sample in progress:
        log_output = trainer.valid_step(sample)

        # log mid-validation stats
        stats = get_valid_stats(trainer)
        for k, v in log_output.items():
            if k in ["loss", "nll_loss", "ntokens", "nsentences", "sample_size"]:
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

    extra_state["tune_eval"]["loss"] = stats["valid_loss"]
    extra_state["tune_eval"]["perplexity"] = stats["valid_ppl"]

    if (
        extra_state["tune_eval"]["lowest_loss"] is None
        or extra_state["tune_eval"]["loss"] < extra_state["tune_eval"]["lowest_loss"]
    ):
        extra_state["tune_eval"]["lowest_loss"] = extra_state["tune_eval"]["loss"]
        extra_state["tune_eval"]["num_since_best"] = 0
    else:
        extra_state["tune_eval"]["num_since_best"] += 1

    stop_due_to_tune_loss = False
    if (
        args.stop_no_best_validate_loss >= 0
        and extra_state["tune_eval"]["num_since_best"] > args.stop_no_best_validate_loss
    ):
        stop_due_to_tune_loss = True
        print(
            f"Stopping training due to eval tune loss stagnation - last best "
            f"eval tune loss of {extra_state['tune_eval']['lowest_loss']} "
            f"(current loss: {extra_state['tune_eval']['loss']}) "
            f"was {extra_state['tune_eval']['num_since_best']} validations ago."
        )
    return extra_state, stop_due_to_tune_loss


def is_training_over_time_limit(extra_state: Dict[str, Any], stop_time: float) -> bool:
    elapsed_hr = (
        time.time() - extra_state["start_time"] + extra_state["previous_training_time"]
    ) / (60 * 60)
    if stop_time >= 0 and elapsed_hr > stop_time:
        print(
            f"Stopping training due to stop time limit of {stop_time} hours - "
            f"we've trained for {elapsed_hr} hours."
        )
        return True
    return False


def save_and_eval(
    args,
    trainer,
    task,
    extra_state: Dict[str, Any],
    checkpoint_manager: Optional[checkpoint.CheckpointManager],
    end_of_epoch=False,
) -> Tuple[Dict[str, Any], bool, Optional[List]]:
    # Checks for time limit stopping criterion even when we're not doing
    # eval/saving checkpoints.
    max_update = args.max_update or math.inf
    stop_due_to_max_update = trainer.get_num_updates() > max_update
    stop_due_to_time_limit = is_training_over_time_limit(extra_state, args.stop_time_hr)
    if not end_of_epoch and (
        args.save_interval_updates <= 0
        or (extra_state["num_iterations"] % args.save_interval_updates != 0)
    ):
        return extra_state, stop_due_to_time_limit

    # Update training time before saving the checkpoint.
    time_now: float = time.time()
    extra_state["previous_training_time"] += time_now - extra_state["start_time"]
    extra_state["start_time"] = time_now

    # Under multiprocessing, each process will run eval over a different
    # shard of the tune data set and then aggregate the results across all
    # processes, so the eval stats from all processes' trainer should
    # remain synchronized.

    # Tune loss
    extra_state, stop_due_to_tune_loss = eval_tune_loss(
        args=args,
        trainer=trainer,
        task=task,
        subset=args.valid_subset,
        extra_state=extra_state,
    )

    is_master: bool = distributed_utils.is_master(args)
    if is_master:
        assert checkpoint_manager is not None, (
            f"Master worker (rank {args.distributed_rank}) should "
            f"have a checkpoint_manager defined."
        )
    else:
        assert checkpoint_manager is None, (
            f"Non-master worker (rank {args.distributed_rank}) should not "
            f"have a checkpoint_manager defined."
        )

    if is_master:
        averaged_params: OrderedDict = checkpoint_manager.get_averaged_params(
            new_params=trainer.get_model().state_dict()
        )
        new_best_averaged_checkpoint = extra_state["tune_eval"]["num_since_best"] == 0
        # checkpoint_manager takes ownership of averaged_params.
        extra_state = checkpoint_manager.save(
            args=args,
            trainer=trainer,
            extra_state=extra_state,
            new_averaged_params=averaged_params,
        )
        if new_best_averaged_checkpoint:
            checkpoint_manager.save_best_averaged_checkpoint(
                args=args, trainer=trainer, extra_state=extra_state
            )

    master_stop_training = None
    if is_master:
        master_stop_training = (
            stop_due_to_time_limit or stop_due_to_tune_loss or stop_due_to_max_update
        )
    stop_training = pytorch_translate_utils.all_gather_from_master(
        args=args, data=[master_stop_training]
    )[0]

    # TODO: fix after masked lm work completes
    if "save_only" not in args or not args.save_only:
        # Basic sanity checks that extra_state is populated correctly.
        assert (
            extra_state["tune_eval"]["loss"] is not None
            and extra_state["tune_eval"]["perplexity"] is not None
        )
    return extra_state, stop_training
