#!/usr/bin/env python3

import argparse


# Helper type for argparse to enable flippable boolean flags. For example,
# group.add_argument("--foo", type=utils.bool_flag, nargs="?", const=True,
#                    default=False)
# creates a --foo flag that defaults to False and can be set to True by
# specifying `--foo True` or just `--foo`. Some other --bar flag that defaults
# to True can be set to False by specifying `--bar False`.
def bool_flag(value):
    if value.lower() in ("true", "t", "1"):
        return True
    elif value.lower() in ("false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Expected boolean string such as 'true'/'false' instead of {value}."
        )
