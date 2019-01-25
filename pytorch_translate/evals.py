#!/usr/bin/env python3

import math
import os
import shutil
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Optional, Tuple

from fairseq import distributed_utils, progress_bar, utils
from fairseq.meters import AverageMeter
from pytorch_translate import (
    checkpoint,
    constants,
    generate,
    utils as pytorch_translate_utils,
)
from pytorch_translate.dual_learning.dual_learning_task import DualLearningTask
from pytorch_translate.tasks.semi_supervised_task import PytorchTranslateSemiSupervised


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
    if trainer.get_meter("loss_scale") is not None:
        stats["loss_scale"] = f"{trainer.get_meter('loss_scale').avg:.3f}"
    stats["wall"] = round(trainer.get_meter("wall").elapsed_time)
    stats["train_wall"] = round(trainer.get_meter("train_wall").sum)
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
        max_sentences=args.max_sentences_valid,
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


def evaluate_bleu(args, task, extra_state, trainer):
    epoch, offset = extra_state["epoch"], extra_state["batch_offset"]
    filename, averaged_state = checkpoint.save_averaged_checkpoint(args, extra_state)
    extra_state["tune_bleu"]["current"], translation_samples = calculate_bleu_on_subset(
        args=args,
        task=task,
        epoch_str=f"{epoch:03d}",
        offset=offset,
        dataset_split=args.valid_subset,
        trainer=trainer,
        model_params=averaged_state["model"],
    )

    if (
        extra_state["tune_bleu"]["best"] is None
        or extra_state["tune_bleu"]["current"] > extra_state["tune_bleu"]["best"]
    ):
        extra_state["tune_bleu"]["best"] = extra_state["tune_bleu"]["current"]
        extra_state["tune_bleu"]["best_epoch"] = epoch
        extra_state["tune_bleu"]["num_since_best"] = 0
        best_filename = os.path.join(
            args.save_dir, constants.AVERAGED_CHECKPOINT_BEST_FILENAME
        )
        shutil.copy2(filename, best_filename)
    else:
        extra_state["tune_bleu"]["num_since_best"] += 1

    stop_due_to_tune_bleu = False
    if (
        args.stop_no_best_bleu_eval >= 0
        and extra_state["tune_bleu"]["num_since_best"] > args.stop_no_best_bleu_eval
    ):
        stop_due_to_tune_bleu = True
        print(
            f"Stopping training due to BLEU score stagnation on tune set - "
            f"last best BLEU score of {extra_state['tune_bleu']['best']} "
            f"(current score: {extra_state['tune_bleu']['current']}) was "
            f"{extra_state['tune_bleu']['num_since_best']} evals ago."
        )
    return extra_state, stop_due_to_tune_bleu, translation_samples


def calculate_bleu_on_subset(
    args,
    task,
    epoch_str: str,
    offset,
    dataset_split,
    trainer,
    model_params: OrderedDict,
):
    # This function constructs a new model object based on the weights to
    # prevent users from accidentally passing in the model from the trainer,
    # since after calling make_generation_fast_(), the model would no longer be
    # suitable for continuing training.
    model = task.build_model(args)
    model.load_state_dict(model_params)

    # This is a trick to have generate use max_sentences_valid
    max_sentences_train = args.max_sentences
    args.max_sentences = args.max_sentences_valid
    datasets = []
    lang_pairs = []
    """
    In multi model training set up, evaluate one model at a time with
    corresponding dataset
    lang_pair is passed to identify model to be used for generation
    """
    if isinstance(task, PytorchTranslateSemiSupervised) or isinstance(
        task, DualLearningTask
    ):
        for key, dataset in task.datasets[dataset_split].datasets.items():
            datasets.append(dataset)
            lang_pairs.append(key)
    else:
        datasets = [task.dataset(dataset_split)]
        lang_pairs = [None]
    score_aggregator_fn = (
        task.score_aggregator if hasattr(task, "score_aggregator") else sum
    )
    scores = []
    for dataset, lang_pair in zip(datasets, lang_pairs):
        # Generate score
        scorer, num_sentences, gen_timer, translation_samples = generate.generate_score(
            args=args, task=task, dataset=dataset, models=[model], lang_pair=lang_pair
        )
        scores.append(scorer.score())
        print(
            f"| epoch {epoch_str} | offset {offset} "
            f"| Eval on {dataset_split} {lang_pair if lang_pair else ''} subset "
            f"with beam={args.beam}: {scorer.result_string()}. "
            f"Generated {num_sentences} sentences ({gen_timer.n} tokens) "
            f"in {gen_timer.sum:.1f}s ({1. / gen_timer.avg:.2f} tokens/s).",
            flush=True,
        )
    # Set max_sentences to its original value
    args.max_sentences = max_sentences_train
    return score_aggregator_fn(scores), translation_samples


def save_and_eval(
    args, trainer, task, extra_state: Dict[str, Any], end_of_epoch=False
) -> Tuple[Dict[str, Any], bool, Optional[list]]:
    # Under multiprocessing, each process will run eval over a different
    # shard of the tune data set and then aggregate the results across all
    # processes, so the eval stats from all processes' trainer should
    # remain synchronized.
    is_master = distributed_utils.is_master(args)

    # Tune loss
    mid_epoch_eval_tune_loss = (args.subepoch_validate_interval > 0) and (
        extra_state["num_iterations"] % args.subepoch_validate_interval == 0
    )
    do_eval_tune_loss = end_of_epoch or mid_epoch_eval_tune_loss
    stop_due_to_tune_loss = False
    if do_eval_tune_loss:
        extra_state, stop_due_to_tune_loss = eval_tune_loss(
            args=args,
            trainer=trainer,
            task=task,
            subset=args.valid_subset,
            extra_state=extra_state,
        )

    # Save
    # Only save checkpoints and eval tune BLEU on the master - all other
    # processes will just get the results from the master.
    mid_epoch_save = (args.save_interval_updates > 0) and (
        extra_state["num_iterations"] % args.save_interval_updates == 0
    )
    end_of_epoch_save = end_of_epoch and not args.no_end_of_epoch_checkpoints
    do_save = is_master and not args.no_save and (mid_epoch_save or end_of_epoch_save)

    if do_save:
        extra_state = checkpoint.save_checkpoint(
            trainer=trainer, args=args, extra_state=extra_state
        )

    # Bleu eval
    mid_epoch_bleu_eval = args.generate_bleu_eval_interval > 0 and (
        extra_state["num_iterations"] - extra_state["tune_bleu"]["last_eval_step"]
        >= args.generate_bleu_eval_interval
    )
    end_of_epoch_bleu_eval = end_of_epoch and args.generate_bleu_eval_per_epoch
    # We can only do BLEU eval when we have a new checkpoint to load.
    do_eval_bleu = (
        is_master and do_save and (mid_epoch_bleu_eval or end_of_epoch_bleu_eval)
    )

    if mid_epoch_bleu_eval and do_eval_bleu:
        extra_state["tune_bleu"]["last_eval_step"] = extra_state["num_iterations"]

    if do_eval_bleu and not do_save:
        raise ValueError(
            "do_save should always be true when do_eval_bleu is true "
            "since a new BLEU eval can only be done when there's a new "
            "checkpoint."
        )

    translation_samples = None
    stop_due_to_tune_bleu = False
    if do_eval_bleu:
        extra_state, stop_due_to_tune_bleu, translation_samples = evaluate_bleu(
            args=args, task=task, extra_state=extra_state, trainer=trainer
        )

    # We don't all_gather the translation_samples since the sample sentences
    # could be pretty long, and only the master uses it anyway.
    master_extra_state = None
    master_stop_training = None
    if is_master:
        master_extra_state = extra_state
        master_stop_training = stop_due_to_tune_loss or stop_due_to_tune_bleu
    extra_state, stop_training = pytorch_translate_utils.all_gather_from_master(
        args=args, data=[master_extra_state, master_stop_training]
    )

    # Basic sanity checks that extra_state is populated correctly.
    assert not (
        do_eval_tune_loss
        and (
            extra_state["tune_eval"]["loss"] is None
            or extra_state["tune_eval"]["perplexity"] is None
        )
    )
    assert not (do_eval_bleu and extra_state["tune_bleu"]["current"] is None)
    return extra_state, stop_training, translation_samples
