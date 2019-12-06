#!/usr/bin/env python3

"""
TODO(T55884145): Deprecate this in favor of using
fvcore.common.file_io.PathManager directly.
"""
from fairseq.file_io import PathManager  # noqa


try:
    from manifold.clients.python import StorageException
except (ImportError, ModuleNotFoundError):

    class StorageException(Exception):
        pass
