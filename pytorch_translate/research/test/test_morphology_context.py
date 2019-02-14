#!/usr/bin/env python3

import unittest

from pytorch_translate.research.unsupervised_morphology import morphology_context


class TestMorphologyContext(unittest.TestCase):
    def test_context_calculations(self):
        contexts = morphology_context.MorphologyContext()
        contexts.add_to_left_context("yeah", 3)
        contexts.add_to_left_context("yeh", 2)
        contexts.add_to_left_context("yeah", 5)
        contexts.add_to_left_context("yeh", 3)

        contexts.add_to_right_context("yeah", 4)
        contexts.add_to_right_context("yeh", 3)
        contexts.add_to_right_context("yeah", 2)
        contexts.add_to_right_context("yeh", 1)

        assert contexts.left_context_count("yeah") == 8
        assert contexts.left_context_count("yeh") == 5
        assert contexts.right_context_count("yeah") == 6
        assert contexts.right_context_count("yeh") == 4

        assert contexts.left_context_sum_count() == 13
        assert contexts.right_context_sum_count() == 10

        assert contexts.left_perplexity() == 1.9469780302238757
        assert contexts.right_perplexity() == 1.9601317042077895
