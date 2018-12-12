#!/usr/bin/env python3

import argparse
import collections
import itertools
import os
from typing import Dict, Optional, Tuple

from fairseq import utils
from pytorch_translate import average_checkpoints, constants


def save_checkpoint(trainer, args, extra_state):
    epoch = extra_state["epoch"]
    batch_offset = extra_state["batch_offset"]
    tune_loss = extra_state["tune_eval"]["loss"]

    if args.log_verbose:
        print(
            f"| Preparing to save checkpoints for epoch {epoch}, "
            f"offset {batch_offset}. ",
            flush=True,
        )

    # batch_offset being None means that we're at the end of an epoch.
    if batch_offset is None:
        if not args.no_epoch_checkpoints:
            epoch_filename = os.path.join(args.save_dir, f"checkpoint{epoch}.pt")
            trainer.save_checkpoint(epoch_filename, extra_state)
            extra_state["last_checkpoints"].append(epoch_filename)

        assert tune_loss is not None

        if (
            extra_state["checkpoint_lowest_loss"] is None
            or tune_loss < extra_state["checkpoint_lowest_loss"]
        ):
            extra_state["checkpoint_lowest_loss"] = tune_loss
            best_filename = os.path.join(args.save_dir, "checkpoint_best.pt")
            trainer.save_checkpoint(best_filename, extra_state)

    # Otherwise, we're in the middle of an epoch.
    elif not args.no_epoch_checkpoints:
        epoch_filename = os.path.join(
            args.save_dir, f"checkpoint{epoch}_{batch_offset}.pt"
        )
        trainer.save_checkpoint(epoch_filename, extra_state)
        extra_state["last_checkpoints"].append(epoch_filename)

    last_filename = os.path.join(args.save_dir, constants.LAST_CHECKPOINT_FILENAME)
    trainer.save_checkpoint(last_filename, extra_state)

    # This ensures we'll always have at least one checkpoint in the list to use
    # for BLEU eval, even if we're not saving epoch checkpoints.
    if args.no_epoch_checkpoints:
        extra_state["last_checkpoints"].append(epoch_filename)
    if args.log_verbose:
        print(
            f"| Finished saving checkpoints for epoch {epoch}, "
            f"offset {batch_offset}.",
            flush=True,
        )
    return extra_state


def save_averaged_checkpoint(args, extra_state):
    epoch, offset = extra_state["epoch"], extra_state["batch_offset"]
    if not hasattr(save_averaged_checkpoint, "last_avg_checkpoints"):
        if args.max_checkpoints_kept == 0:
            raise argparse.ArgumentTypeError("--max-checkpoints-kept must be != 0.")
        save_averaged_checkpoint.last_avg_checkpoints = ManagedCheckpoints(
            max(args.max_checkpoints_kept, 1), auto_clear=args.max_checkpoints_kept > 0
        )

    last_checkpoints = extra_state["last_checkpoints"].get_last_n(
        1 if args.no_epoch_checkpoints else args.generate_bleu_eval_avg_checkpoints
    )
    if args.log_verbose:
        print(
            f"| Reading {len(last_checkpoints)} previous "
            f"checkpoints for averaging in epoch {epoch}, offset {offset}.",
            flush=True,
        )
    averaged_state = average_checkpoints.average_checkpoints(last_checkpoints)
    filename = os.path.join(args.save_dir, f"averaged_checkpoint{epoch}_{offset}.pt")
    save_averaged_checkpoint.last_avg_checkpoints.append(filename)
    if args.log_verbose:
        print(
            f"| Preparing to save averaged checkpoint for "
            f"epoch {epoch}, offset {offset}.",
            flush=True,
        )
    utils.torch_persistent_save(averaged_state, filename)
    if args.log_verbose:
        print(
            f"| Finished saving averaged checkpoint for "
            f"epoch {epoch}, offset {offset}.",
            flush=True,
        )
    return filename


def load_existing_checkpoint(
    checkpoint_path, trainer, restore_state=True
) -> Tuple[bool, Optional[Dict]]:
    loaded = False
    extra_state = None

    if not os.path.isfile(checkpoint_path):
        print(
            f"| No existing checkpoint at {checkpoint_path}. "
            f"Starting training from scratch."
        )
        return loaded, extra_state

    if restore_state:
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is None:
            loaded = False
            print(f"| Failed to load checkpoint and state from {checkpoint_path}.")
        else:
            loaded = True
            print(
                f"| Loaded checkpoint {checkpoint_path} "
                f"(epoch {extra_state['epoch']}) with restored extra state."
            )
            # batch_offset being None denotes this was a checkpoint saved at
            # the end of an epoch (after the last batch).
            if extra_state["batch_offset"] is None:
                trainer.lr_step(extra_state["epoch"])
                extra_state["epoch"] += 1
                extra_state["batch_offset"] = 0

    else:
        dummy_state = trainer.load_checkpoint(checkpoint_path, reset_optimizer=True)
        if dummy_state is None:
            loaded = False
            print(f"| Failed to load checkpoint weights from {checkpoint_path}.")
        else:
            loaded = True
            print(f"| Loaded checkpoint weights from {checkpoint_path}.")

    return loaded, extra_state


# The purpose of this class is to keep track of the list of checkpoints
# currently alive and automatically delete those that are no more required
# and that we do not want to keep around.
# In a nutshell, this class remembers the last max_num_checkpoints
# and delete (auto_clear == True) the oldest checkpoint each time a new one
# is added past this number.
# TODO(T34212782): replace this class with simple list of strings and helper fns
# to avoid messy pickling/unpickling of objects.
class ManagedCheckpoints:

    # - max_num_checkpoints: Maximum number of checkpoints we need at one point.
    # - auto_clear: Control whether or not checkpoints should get deleted when
    #   they are not in the last max_num_checkpoints appended to the
    #   self anymore.
    def __init__(self, max_num_checkpoints, auto_clear):
        self.auto_clear = auto_clear
        assert max_num_checkpoints > 0, "Empty listing is not supported"
        self.kept_checkpoints = collections.deque(maxlen=max_num_checkpoints)

    def __eq__(self, other):
        return (
            isinstance(other, ManagedCheckpoints)
            and self.auto_clear == other.auto_clear
            and self.kept_checkpoints == other.kept_checkpoints
        )

    def __repr__(self):
        return (
            f"ManagedCheckpoints(kept_checkpoints={self.kept_checkpoints}, "
            f"auto_clear={self.auto_clear})"
        )

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
        assert not self.auto_clear or not self.kept_checkpoints.count(
            checkpoint_filename
        ), "Not yet implemented"
        if (
            self.auto_clear
            and len(self.kept_checkpoints) == self.kept_checkpoints.maxlen
        ):
            # We reach the max number of checkpoints we keep around.
            # Delete the oldest one.
            try:
                os.remove(self.kept_checkpoints.popleft())
            except FileNotFoundError:
                pass
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
