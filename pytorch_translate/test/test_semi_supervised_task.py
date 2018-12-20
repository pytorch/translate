#!/usr/bin/env python3

import unittest
from itertools import zip_longest

from pytorch_translate.tasks.semi_supervised_task import PytorchTranslateSemiSupervised


class TestSemiSupervisedTask(unittest.TestCase):
    def test_parse_loss_weights(self):
        """ Make sure we can decode a loss_weights json. """
        loss_weights_json = """[
            [5, {'src-tgt': 1, 'src-tgt_mono': 0, 'tgt-src': 1, 'tgt-src_mono': 0}],
            [5, {'src-tgt': 1, 'src-tgt_mono': 0.5, 'tgt-src': 1, 'tgt-src_mono': 0.5}],
            [100, {'src-tgt': 1, 'src-tgt_mono': 1, 'tgt-src': 1, 'tgt-src_mono': 1}]
        ]"""
        expected_loss_weights = [
            (5, {"src-tgt": 1, "src-tgt_mono": 0, "tgt-src": 1, "tgt-src_mono": 0}),
            (5, {"src-tgt": 1, "src-tgt_mono": 0.5, "tgt-src": 1, "tgt-src_mono": 0.5}),
            (100, {"src-tgt": 1, "src-tgt_mono": 1, "tgt-src": 1, "tgt-src_mono": 1}),
        ]
        parsed_loss_weights = PytorchTranslateSemiSupervised.parse_loss_weights(
            loss_weights_json=loss_weights_json
        )
        for parsed_schedule, expected_schedule in zip_longest(
            parsed_loss_weights, expected_loss_weights
        ):
            self.assertEquals(parsed_schedule[0], expected_schedule[0])
            self.assertDictEqual(parsed_schedule[1], expected_schedule[1])
