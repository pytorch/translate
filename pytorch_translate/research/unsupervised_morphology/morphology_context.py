#!/usr/bin/env python3

import math
from collections import Counter


class MorphologyContext:
    """
    This class is a data holder for keeping left and right context counts for
    morphology. This is inspired from the following paper:
    https://users.ics.aalto.fi/krista/papers/Creutz07.pdf
    """

    def __init__(self):
        self.left_context = Counter()
        self.right_context = Counter()

    def add_to_left_context(self, substr, count):
        self.left_context[substr] += count

    def add_to_right_context(self, substr, count):
        self.right_context[substr] += count

    def left_context_count(self, substr):
        return self.left_context[substr]

    def right_context_count(self, substr):
        return self.right_context[substr]

    def left_context_sum_count(self):
        return sum(self.left_context.values())

    def right_context_sum_count(self):
        return sum(self.right_context.values())

    def perplexity(self, direction: str):
        contexts = self.left_context if direction == "left" else self.right_context
        context_count = (
            self.left_context_sum_count()
            if direction == "left"
            else self.right_context_sum_count()
        )
        entropy = 0
        for context in contexts:
            prob = float(contexts[context]) / context_count
            entropy -= prob * math.log(prob)
        return math.exp(entropy)

    def left_perplexity(self):
        return self.perplexity("left")

    def right_perplexity(self):
        return self.perplexity("right")
