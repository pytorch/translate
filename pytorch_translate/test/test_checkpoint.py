#!/usr/bin/env python3

import copy
import itertools
import os
import tempfile
import unittest
from collections import OrderedDict

import numpy as np
import torch
from pytorch_translate import checkpoint
from pytorch_translate.checkpoint import CheckpointManager
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


class CheckpointManagerTest(unittest.TestCase):
    def setUp(self):
        self._params_1 = OrderedDict(
            [
                ("double_tensor", torch.DoubleTensor([100.0])),
                ("float_tensor", torch.FloatTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                ("long_tensor", torch.LongTensor([7, 8, 9])),
                ("half_tensor", torch.HalfTensor([10.0, 20.0])),
            ]
        )
        self._params_2 = OrderedDict(
            [
                ("double_tensor", torch.DoubleTensor([1.0])),
                ("float_tensor", torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])),
                # Any integer tensor must remain the same in all the params.
                ("long_tensor", torch.LongTensor([7, 8, 9])),
                ("half_tensor", torch.HalfTensor([50.0, 0.0])),
            ]
        )
        self._params_avg = OrderedDict(
            [
                ("double_tensor", torch.DoubleTensor([50.5])),
                ("float_tensor", torch.FloatTensor([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]])),
                ("long_tensor", torch.LongTensor([7, 8, 9])),
                # We convert fp16 to fp32 when averaging params.
                ("half_tensor", torch.FloatTensor([30.0, 10.0])),
            ]
        )

        self._fd_1, self._filename_1 = tempfile.mkstemp()
        self._fd_2, self._filename_2 = tempfile.mkstemp()
        torch.save(OrderedDict([("model", self._params_1)]), self._filename_1)
        torch.save(OrderedDict([("model", self._params_2)]), self._filename_2)

    def tearDown(self):
        os.close(self._fd_1)
        os.remove(self._filename_1)
        os.close(self._fd_2)
        os.remove(self._filename_2)

    def _check_params(self, expected_params: OrderedDict, actual_params: OrderedDict):
        for (k_expected, v_expected), (k_actual, v_actual) in itertools.zip_longest(
            expected_params.items(), actual_params.items()
        ):
            self.assertEqual(
                k_expected,
                k_actual,
                (
                    f"Key mismatch - expected {k_expected} but found {k_actual}. "
                    f"(Expected list of keys: {expected_params.keys()} "
                    f"vs actual list of keys: {actual_params.keys()})"
                ),
            )
            np.testing.assert_allclose(
                v_expected.numpy(),
                v_actual.numpy(),
                err_msg=f"Tensor value mismatch for key {k_expected}",
            )

    def test_initialize(self):
        checkpoint_manager = CheckpointManager(
            num_avg_checkpoints=2,
            auto_clear_checkpoints=False,
            log_verbose=False,
            checkpoint_files=[self._filename_1, self._filename_2],
        )
        checkpoint_manager._initialize()
        self._check_params(
            expected_params=self._params_avg,
            actual_params=checkpoint_manager._averaged_params,
        )

        # Tests providing 3 old checkpoint files while only using
        # num_avg_checkpoints=2. The oldest one should not be included in the
        # average.
        checkpoint_manager = CheckpointManager(
            num_avg_checkpoints=2,
            auto_clear_checkpoints=False,
            log_verbose=False,
            checkpoint_files=[self._filename_1, self._filename_1, self._filename_2],
        )
        checkpoint_manager._initialize()
        self._check_params(
            expected_params=self._params_avg,
            actual_params=checkpoint_manager._averaged_params,
        )

    def test_get_averaged_params(self):
        checkpoint_manager = CheckpointManager(
            num_avg_checkpoints=2,
            auto_clear_checkpoints=True,
            log_verbose=False,
            checkpoint_files=[],
        )

        # The first average should be the same as the first params.
        avg_params = checkpoint_manager.get_averaged_params(new_params=self._params_1)
        self._check_params(expected_params=self._params_1, actual_params=avg_params)
        checkpoint_to_remove = checkpoint_manager._update_state(
            new_params_filename=self._filename_1,
            new_averaged_params=avg_params,
        )
        self.assertEqual(None, checkpoint_to_remove)
        self._check_params(
            expected_params=self._params_1,
            actual_params=checkpoint_manager._averaged_params,
        )

        # Adding the second params should produce the expected average.
        avg_params = checkpoint_manager.get_averaged_params(new_params=self._params_2)
        self._check_params(expected_params=self._params_avg, actual_params=avg_params)
        checkpoint_to_remove = checkpoint_manager._update_state(
            new_params_filename=self._filename_2,
            new_averaged_params=avg_params,
        )
        self.assertEqual(None, checkpoint_to_remove)
        self._check_params(
            expected_params=self._params_avg,
            actual_params=checkpoint_manager._averaged_params,
        )

        # Adding the second params again should produce an average that's the
        # same as the second params.
        avg_params = checkpoint_manager.get_averaged_params(new_params=self._params_2)
        self._check_params(expected_params=self._params_2, actual_params=avg_params)
        checkpoint_to_remove = checkpoint_manager._update_state(
            new_params_filename=self._filename_2,
            new_averaged_params=avg_params,
        )
        # This should kick out the first params file.
        self.assertEqual(self._filename_1, checkpoint_to_remove)
        self._check_params(
            expected_params=self._params_2,
            actual_params=checkpoint_manager._averaged_params,
        )

    def test_integer_tensor_change_error(self):
        params_invalid = copy.deepcopy(self._params_1)
        # An integer tensor is expected to remain constant and should not change
        # its value.
        params_invalid["long_tensor"] += 1
        fd_invalid, filename_invalid = tempfile.mkstemp()
        torch.save(OrderedDict([("model", params_invalid)]), filename_invalid)

        # Tests trying to initialize with an invalid param.
        with self.assertRaises(ValueError):
            checkpoint_manager = CheckpointManager(
                num_avg_checkpoints=2,
                auto_clear_checkpoints=False,
                log_verbose=False,
                checkpoint_files=[self._filename_1, filename_invalid],
            )
            checkpoint_manager._initialize()

        # Tests trying to get the average with an invalid new param when the
        # queue isn't full.
        checkpoint_manager = CheckpointManager(
            num_avg_checkpoints=2,
            auto_clear_checkpoints=False,
            log_verbose=False,
            checkpoint_files=[self._filename_1],
        )
        with self.assertRaises(ValueError):
            checkpoint_manager.get_averaged_params(new_params=params_invalid)

        # Test trying to get the average with an invalid new param when the
        # queue is full.
        checkpoint_manager = CheckpointManager(
            num_avg_checkpoints=2,
            auto_clear_checkpoints=False,
            log_verbose=False,
            checkpoint_files=[self._filename_1, self._filename_2],
        )
        with self.assertRaises(ValueError):
            checkpoint_manager.get_averaged_params(new_params=params_invalid)

        os.close(fd_invalid)
        os.remove(filename_invalid)
