#!/usr/bin/env python3

import importlib
import os


# automatically import any Python files in the tasks/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        task_name = file[: file.find(".py")]
        importlib.import_module("pytorch_translate.tasks." + task_name)
