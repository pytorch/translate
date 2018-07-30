#!/usr/bin/env python3

import collections
from typing import Any, Dict, Iterable, List

import torch


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
            summed_v = summed_v + x if summed_v is not None else x
        if summed_v is not None:
            averaged_params[k] = summed_v / len(v)
    new_state["model"] = averaged_params
    return new_state
