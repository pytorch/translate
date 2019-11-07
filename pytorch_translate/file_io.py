#!/usr/bin/env python3

"""
TODO(T55884145): Deprecate this in favor of using
fvcore.common.file_io.PathManager directly.
"""
import os
from typing import List


try:
    from fvcore.common.file_io import PathManager

except (ImportError, ModuleNotFoundError):

    class PathManager:
        @staticmethod
        def open(path: str, mode: str = "r"):
            return open(path, mode)

        @staticmethod
        def copy(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
            raise NotImplementedError()

        @staticmethod
        def get_local_path(path: str) -> str:
            raise NotImplementedError()

        @staticmethod
        def exists(path: str) -> bool:
            return os.path.exists(path)

        @staticmethod
        def isfile(path: str) -> bool:
            return os.path.isfile(path)

        @staticmethod
        def ls(path: str) -> List[str]:
            return os.listdir(path)

        @staticmethod
        def mkdirs(path: str):
            os.makedirs(path, exist_ok=True)

        @staticmethod
        def rm(path: str):
            os.remove(path)
