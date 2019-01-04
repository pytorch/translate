#!/usr/bin/env python3

import os
import unittest

import torch
from pytorch_translate import checkpoint
from pytorch_translate.test import utils as test_utils


class TestCheckpoint(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_load_checkpoint(self):
        """Train for one step, save a checkpoint, and make sure it is loaded
        properly."""
        test_save_file = test_utils.make_temp_file()
        test_args = test_utils.ModelParamsDict()
        test_args.distributed_rank = 0
        extra_state = test_utils.create_dummy_extra_state(epoch=2)
        trainer, _ = test_utils.gpu_train_step(test_args)
        trainer.save_checkpoint(test_save_file, extra_state)
        loaded, extra_state = checkpoint.load_existing_checkpoint(
            test_save_file, trainer, restore_state=True
        )
        # Loading checkpoint without restore state should reset extra state
        assert loaded and extra_state["epoch"] == 2
        os.remove(test_save_file)

    @unittest.skipIf(torch.cuda.device_count() < 1, "No GPU available for test.")
    def test_load_checkpoint_no_restore_state(self):
        """Train for one step, save a checkpoint, and make sure it is loaded
        properly WITHOUT loading the extra state from the checkpoint."""
        test_save_file = test_utils.make_temp_file()
        test_args = test_utils.ModelParamsDict()
        test_args.distributed_rank = 0
        extra_state = test_utils.create_dummy_extra_state(epoch=2)
        trainer, _ = test_utils.gpu_train_step(test_args)
        trainer.save_checkpoint(test_save_file, extra_state)
        loaded, extra_state = checkpoint.load_existing_checkpoint(
            test_save_file, trainer, restore_state=False
        )
        # Loading checkpoint without restore state should reset extra state
        assert loaded and extra_state is None
        os.remove(test_save_file)
