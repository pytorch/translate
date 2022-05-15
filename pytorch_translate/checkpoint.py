#!/usr/bin/env python3

import os
from collections import deque, OrderedDict
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
from fairseq import checkpoint_utils, utils as fairseq_utils
from pytorch_translate import constants

# TODO(T55884145): Replace with
# from fvcore.common.file_io import PathManager
from pytorch_translate.file_io import PathManager


def save_checkpoint_atomic(trainer, final_filename, extra_state):
    """Wrapper around trainer.save_checkpoint to make save atomic."""
    temp_filename = os.path.join(final_filename + ".tmp")
    trainer.save_checkpoint(temp_filename, extra_state)
    # TODO(T56266125): Use mv() instead of copy() + rm() after it's added to
    # PathManager.
    assert PathManager.copy(
        temp_filename, final_filename, overwrite=True
    ), f"Failed to copy {temp_filename} to {final_filename}"
    PathManager.rm(temp_filename)


def load_existing_checkpoint(
    checkpoint_path, trainer, restore_state=True
) -> Tuple[bool, Optional[Dict]]:
    loaded = False
    extra_state = None

    if not PathManager.isfile(checkpoint_path):
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


def load_to_cpu(path: str) -> Dict[str, Any]:
    """
    This is just fairseq's utils.load_checkpoint_to_cpu(), except we don't try
    to upgrade the state dict for backward compatibility - to make cases
    where we only care about loading the model params easier to unit test.
    """
    with PathManager.open(path, "rb") as f:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )
    return state


def load_to_gpu(path: str) -> Dict[str, Any]:
    """
    Similar to load_to_cpu, but load model to cuda
    """
    with PathManager.open(path, "rb") as f:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cuda")
            ),
        )
    return state


def is_integer_tensor(tensor: torch.Tensor) -> bool:
    return (
        isinstance(tensor, torch.ByteTensor)
        or isinstance(tensor, torch.CharTensor)
        or isinstance(tensor, torch.ShortTensor)
        or isinstance(tensor, torch.IntTensor)
        or isinstance(tensor, torch.LongTensor)
    )


def sanity_check_tensor(
    tensor_name: str, old_tensor: torch.Tensor, new_tensor: torch.Tensor
) -> None:
    if old_tensor.type() != new_tensor.type():
        raise ValueError(
            f"Type mismatch for tensor {tensor_name}. Old tensor had type of "
            f"{old_tensor.type()} while new tensor has type of {new_tensor.type()}."
        )
    if old_tensor.size() != new_tensor.size():
        raise ValueError(
            f"Size mismatch for tensor {tensor_name} of type {old_tensor.type()}. "
            f"Old tensor had size of {old_tensor.size()} while new tensor "
            f"has size of {new_tensor.size()}."
        )

    if is_integer_tensor(old_tensor):
        # The following sanity check is only relevant for integer tensors - which
        # we expect to be index-like, and therefore should remain constant and not
        # be averaged over.
        if not torch.all(old_tensor == new_tensor):
            raise ValueError(
                f"Integer tensor {tensor_name} of type {old_tensor.type()} "
                f"and size {old_tensor.size()} had "
                f"{torch.sum(old_tensor != new_tensor).item()} mismatched elements."
            )


def convert_tensor(tensor: torch.Tensor, clone: bool) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if isinstance(tensor, torch.HalfTensor):
        # We convert any fp16 params to fp32 to make sure operations like
        # division by a scalar value are supported.
        tensor = tensor.float()
    elif clone:
        # tensor.float() would have effectively cloned the fp16 tensor already,
        # so we don't need to do it again even if clone=True.
        tensor = tensor.clone()
    return tensor


class CheckpointManager:
    """Class to help manage, save, clean up, and average checkpoints."""

    def __init__(
        self,
        num_avg_checkpoints: int,
        auto_clear_checkpoints: bool,
        log_verbose: bool,
        checkpoint_files: List[str],
    ):
        """
        Args:
          num_avg_checkpoints: Number of checkpoints to average over.
          auto_clear_checkpoints: If True, we automatically delete
              checkpoints older than args.num_avg_checkpoints.
          log_verbose:
        """
        assert num_avg_checkpoints > 0, "Must average over at least one checkpoint."
        self._num_avg_checkpoints: int = num_avg_checkpoints
        self._auto_clear_checkpoints: bool = auto_clear_checkpoints
        self._log_verbose: bool = log_verbose
        self._averaged_params: OrderedDict = OrderedDict()
        self._checkpoint_files: Deque[str] = deque(maxlen=self._num_avg_checkpoints)
        # Updates the checkpoint files deque and discards any checkpoint files
        # older than the limit.
        for file in checkpoint_files:
            self._checkpoint_files.append(file)
        # Defers actually reading the checkpoint files until later due to
        # T39501955.
        self._initialized = False

    def __repr__(self):
        return (
            f"CheckpointManager(num_avg_checkpoints={self._num_avg_checkpoints}, "
            f"auto_clear_checkpoints={self._auto_clear_checkpoints}, "
            f"log_verbose={self._log_verbose}, "
            f"checkpoint_files={self._checkpoint_files})"
        )

    def _initialize(self):
        # Loads and intializes the previous checkpoint params average.
        for f in self._checkpoint_files:
            # Loads everything to CPU memory to save space on GPU memory.
            state: Dict[str, Any] = load_to_cpu(f)
            model_params: OrderedDict = state["model"]
            for k, v in model_params.items():
                v = convert_tensor(v, clone=False)
                if k not in self._averaged_params:
                    self._averaged_params[k] = (
                        v if is_integer_tensor(v) else v / len(self._checkpoint_files)
                    )
                else:
                    sanity_check_tensor(
                        tensor_name=k, old_tensor=self._averaged_params[k], new_tensor=v
                    )
                    if not is_integer_tensor(v):
                        self._averaged_params[k].add_(v / len(self._checkpoint_files))

        self._initialized = True

    def log_if_verbose(self, msg: str):
        if self._log_verbose:
            print(msg, flush=True)

    def get_averaged_params(self, new_params: OrderedDict) -> OrderedDict:
        if not self._initialized:
            self._initialize()

        self.log_if_verbose(f"| Preparing to average {len(new_params)} params.")
        # Special case for the first time or when we're not doing checkpoint
        # averaging.
        if len(self._averaged_params) == 0 or self._checkpoint_files.maxlen == 1:
            copied_params: OrderedDict = OrderedDict()
            for k, v in new_params.items():
                copied_params[k] = convert_tensor(v, clone=True)
            self.log_if_verbose(
                f"| Finished copying {len(new_params)} params to CPU "
                f"(no averaging needed)."
            )
            return copied_params

        new_average: OrderedDict = OrderedDict()
        new_denom: int = min(  # noqa
            self._checkpoint_files.maxlen, len(self._checkpoint_files) + 1
        )

        if len(self._checkpoint_files) == self._checkpoint_files.maxlen:
            # We've reached the maximum number of checkpoints to average over,
            # so the denominator won't change even when we add a new param - we
            # just kick out the values from the oldest checkpoint.
            self.log_if_verbose(
                f"| Preparing to load old checkpoint "
                f"{self._checkpoint_files[0]} to calculate average."
            )
            state: Dict[str, Any] = load_to_cpu(self._checkpoint_files[0])
            self.log_if_verbose(
                f"| Finished loading old checkpoint "
                f"{self._checkpoint_files[0]} to calculate average."
            )
            old_params: OrderedDict = state["model"]
            for k, v in old_params.items():
                v = convert_tensor(v, clone=False)
                sanity_check_tensor(
                    tensor_name=k, old_tensor=self._averaged_params[k], new_tensor=v
                )
                if is_integer_tensor(v):
                    new_average[k] = v
                else:
                    new_average[k] = self._averaged_params[k] - (
                        v / len(self._checkpoint_files)
                    )

        else:
            # We haven't reached the maximum number of checkpoints to average
            # over, so we simply adjust the denominator and the existing average
            # to account for the larger number of checkpoints we're now
            # averaging over.
            for k, v in self._averaged_params.items():
                if is_integer_tensor(v):
                    new_average[k] = v.clone()
                else:
                    new_average[k] = v * (len(self._checkpoint_files) / new_denom)

        # Actually add the new params after the existing average has been
        # adjusted accordingly.
        for k, v in new_params.items():
            v = convert_tensor(v, clone=False)
            sanity_check_tensor(tensor_name=k, old_tensor=new_average[k], new_tensor=v)
            if not is_integer_tensor(v):
                new_average[k].add_(v / new_denom)

        self.log_if_verbose(f"| Finished averaging {len(new_params)} params.")
        return new_average

    def _update_state(
        self, new_params_filename: str, new_averaged_params: OrderedDict
    ) -> Optional[str]:
        # Consider making a copy of each tensor here if we run into issues in
        # the future with callers later modifying the params passed in.
        self._averaged_params = new_averaged_params

        checkpoint_to_remove = None
        if (
            self._auto_clear_checkpoints
            and len(self._checkpoint_files) == self._checkpoint_files.maxlen
        ):
            # We delay actually removing this checkpoint until after the newest
            # checkpoint has been successfully written.
            checkpoint_to_remove = self._checkpoint_files.popleft()

        # Make sure to include the checkpoint itself in its list of checkpoint
        # files - this is to ensure we can still restore everything correctly
        # even if the file gets copied to another name (ex: checkpoint_last.py).
        self._checkpoint_files.append(new_params_filename)
        return checkpoint_to_remove

    def _remove_checkpoint(self, checkpoint_to_remove: Optional[str]):
        if checkpoint_to_remove:
            self.log_if_verbose(
                f"| Preparing to remove old checkpoint {checkpoint_to_remove}."
            )
            try:
                PathManager.rm(checkpoint_to_remove)
                self.log_if_verbose(
                    f"| Finished removing old checkpoint {checkpoint_to_remove}."
                )
            except OSError as e:
                print(
                    f"| Failed to remove old checkpoint {checkpoint_to_remove} "
                    f"- exception: {e}",
                    flush=True,
                )

    def remove_all_checkpoints(self):
        """
        Removes all checkpoints besides
         average_checkpoint.pt and last_checkpoint.pt
        """
        for checkpoint_to_remove in self._checkpoint_files:
            self._remove_checkpoint(checkpoint_to_remove)

    def save(
        self,
        args,
        trainer,
        extra_state: Dict[str, Any],
        new_averaged_params: OrderedDict,
    ) -> Dict[str, Any]:
        """Saves the model params contained in trainer.

        Takes ownership of new_averaged_params, so the caller should not modify
        them afterwards.

        Args:
          trainer: Trainer containing the model to be saved.
          extra_state: Dictionary containing any extra information about the
              model beyond the param weights.
          new_averaged_params: If specified, takes ownership of the params and
              sets them as current set of averaged params. If not specified,
              we will recalculate the averaged params using the model params
              in trainer.

        Returns:
          Updated extra_state dictionary.
        """
        epoch = extra_state["epoch"]
        batch_offset = extra_state["batch_offset"]

        # batch_offset being None means that we're at the end of an epoch.
        if batch_offset is None:
            filename = os.path.join(args.save_dir, f"checkpoint{epoch}_end.pt")
        # Otherwise, we're in the middle of an epoch.
        else:
            filename = os.path.join(
                args.save_dir, f"checkpoint{epoch}_{batch_offset}.pt"
            )

        checkpoint_to_remove = self._update_state(
            new_params_filename=filename, new_averaged_params=new_averaged_params
        )
        extra_state["checkpoint_files"] = list(self._checkpoint_files)

        self.log_if_verbose(
            f"| Preparing to save checkpoints for epoch {epoch}, "
            f"offset {batch_offset}."
        )
        # Saves two copies of the checkpoint - one under a specific name
        # corresponding to its epoch/offset, and another under the generic
        # "checkpoint_last.py" that we restore from in case training is
        # interrupted.
        save_checkpoint_atomic(
            trainer=trainer, final_filename=filename, extra_state=extra_state
        )
        # We update checkpoint_last.pt only after the new averaged checkpoint
        # and epoch/offset-named copy have been written - so that in case either
        # write fails, we'd still be able to resume from the previous
        # checkpoint_last.pt
        last_checkpoint_path = os.path.join(
            args.save_dir, constants.LAST_CHECKPOINT_FILENAME
        )
        assert PathManager.copy(
            filename, last_checkpoint_path, overwrite=True
        ), f"Failed to copy {filename} to {last_checkpoint_path}"
        self.log_if_verbose(
            f"| Finished saving checkpoints for epoch {epoch}, "
            f"offset {batch_offset}."
        )

        # Wait until after checkpoint_last.py has been written to remove the
        # oldest checkpoint. This is so that in case we fail to write a new
        # checkpoint_last.py, we'd still have access to all the files listed in
        # the previous checkpoint_last.py
        self._remove_checkpoint(checkpoint_to_remove)
        return extra_state

    def save_best_averaged_checkpoint(self, args, trainer, extra_state: Dict[str, Any]):
        """
        save() should always be called before calling this function - to ensure
        that extra_state and self._averaged_params have been updated correctly.
        """
        best_averaged_checkpoint_filename = os.path.join(
            args.save_dir, constants.AVERAGED_CHECKPOINT_BEST_FILENAME
        )
        self.log_if_verbose(
            f"| Preparing to save new best averaged checkpoint to "
            f"{best_averaged_checkpoint_filename}."
        )
        state_dict = trainer.state_dict()
        state_dict["args"] = args
        state_dict["cfg"] = None
        state_dict["model"] = self._averaged_params
        state_dict["extra_state"].update(extra_state)
        state_dict = fairseq_utils.move_to_cpu(state_dict)
        checkpoint_utils.torch_persistent_save(
            obj=state_dict,
            filename=best_averaged_checkpoint_filename,
        )
        self.log_if_verbose(
            f"| Finished saving new best averaged checkpoint to "
            f"{best_averaged_checkpoint_filename}."
        )
