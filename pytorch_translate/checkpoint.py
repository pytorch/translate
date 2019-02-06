#!/usr/bin/env python3

import argparse
import collections
import itertools
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from fairseq import utils
from pytorch_translate import constants


def save_checkpoint_atomic(trainer, final_filename, extra_state):
    """Wrapper around trainer.save_checkpoint to make save atomic."""
    path, filename = os.path.split(final_filename)
    temp_filename = os.path.join(path, "." + filename + ".tmp")

    trainer.save_checkpoint(temp_filename, extra_state)
    os.rename(temp_filename, final_filename)


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
            save_checkpoint_atomic(trainer, epoch_filename, extra_state)
            extra_state["last_checkpoints"].append(epoch_filename)

        assert tune_loss is not None

        if (
            extra_state["checkpoint_lowest_loss"] is None
            or tune_loss < extra_state["checkpoint_lowest_loss"]
        ):
            extra_state["checkpoint_lowest_loss"] = tune_loss
            best_filename = os.path.join(args.save_dir, "checkpoint_best.pt")
            save_checkpoint_atomic(trainer, best_filename, extra_state)

    # Otherwise, we're in the middle of an epoch.
    elif not args.no_epoch_checkpoints:
        epoch_filename = os.path.join(
            args.save_dir, f"checkpoint{epoch}_{batch_offset}.pt"
        )
        save_checkpoint_atomic(trainer, epoch_filename, extra_state)
        extra_state["last_checkpoints"].append(epoch_filename)

    last_filename = os.path.join(args.save_dir, constants.LAST_CHECKPOINT_FILENAME)
    save_checkpoint_atomic(trainer, last_filename, extra_state)

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


def average_checkpoints(inputs: Iterable[str]) -> Dict[str, Any]:
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict: collections.OrderedDict = collections.OrderedDict()
    params_keys: List = []
    new_state: Dict[str, Any] = {}
    for f in inputs:
        state: Dict[str, Any] = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )
        # We want to use the extra_state and other params from the last
        # (most recent) checkpoint.
        new_state = state

        model_params: collections.OrderedDict = state["model"]

        model_params_keys = list(model_params.keys())
        if len(params_keys) == 0:
            params_keys = model_params_keys
        if params_keys != model_params_keys:
            raise KeyError(
                f"For checkpoint {f}, expected list of params: {params_keys}, "
                f"but found: {model_params_keys}"
            )

        for k in params_keys:
            if k not in params_dict:
                params_dict[k] = []
            params_dict[k].append(model_params[k])
    averaged_params: collections.OrderedDict = collections.OrderedDict()
    # v should be a list of torch Tensor.
    for k, v in params_dict.items():
        summed_v = None
        for x in v:
            if isinstance(x, torch.HalfTensor):
                x = x.float()
            summed_v = summed_v + x if summed_v is not None else x
        if summed_v is not None:
            averaged_params[k] = summed_v / len(v)
    new_state["model"] = averaged_params
    return new_state


def save_averaged_checkpoint(args, extra_state):
    epoch, offset = extra_state["epoch"], extra_state["batch_offset"]
    if not hasattr(save_averaged_checkpoint, "last_avg_checkpoints"):
        if args.max_checkpoints_kept == 0:
            raise argparse.ArgumentTypeError("--max-checkpoints-kept must be != 0.")
        save_averaged_checkpoint.last_avg_checkpoints = ManagedCheckpoints(
            max(args.max_checkpoints_kept, 1), auto_clear=args.max_checkpoints_kept > 0
        )

    last_checkpoints = extra_state["last_checkpoints"].get_last_n(
        1 if args.no_epoch_checkpoints else args.num_avg_checkpoints
    )
    if args.log_verbose:
        print(
            f"| Reading {len(last_checkpoints)} previous "
            f"checkpoints for averaging in epoch {epoch}, offset {offset}.",
            flush=True,
        )
    averaged_state = average_checkpoints(last_checkpoints)
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
    return filename, averaged_state


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
