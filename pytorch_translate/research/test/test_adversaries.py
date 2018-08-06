#!/usr/bin/env python3

import unittest

import torch
import torch.nn.functional as F

from pytorch_translate.dictionary import Dictionary
from pytorch_translate.tasks import PytorchTranslateTask
from pytorch_translate.research.adversarial import adversaries

from typing import NamedTuple

# For determinism
torch.manual_seed(11051993)


class ToyEncoder(NamedTuple):
    embed_tokens: torch.nn.Embedding


class ToyModel(NamedTuple):
    encoder: ToyEncoder


class DummyArgs(NamedTuple):
    max_swaps: int
    normalize_directions: bool


class AdversariesTest(unittest.TestCase):
    def setUp(self):
        self.num_random_retries = 10
        self.voc_size = 4
        self.dim = 5
        self.bsz = 3
        self.len = 6
        self.model = self._toy_model()
        self.embed_tokens = self.model.encoder.embed_tokens
        self.eta = 0.001

    def _toy_model(self):
        embed_tokens = torch.nn.Embedding(self.voc_size, self.dim)
        embed_tokens.weight.data = torch.eye(self.voc_size, self.dim)
        model = ToyModel(encoder=ToyEncoder(embed_tokens=embed_tokens))
        return model

    def _toy_task(self, args):
        src_dict = Dictionary()
        symbols = "abcdefghijklmnopqrstuvwxyz"
        for symbol in symbols[: self.voc_size]:
            src_dict.add_symbol(symbol)
        dst_dict = Dictionary()
        return PytorchTranslateTask(args, src_dict, dst_dict)

    def _get_brute_force_adversary(self, max_swaps=1, normalize_directions=False):
        args = DummyArgs(
            max_swaps=max_swaps, normalize_directions=normalize_directions
        )
        adv = adversaries.brute_force_adversary.BruteForceAdversary(
            args, self.model, self._toy_task(args)
        )
        return adv

    def _dummy_gradient(self):
        return torch.rand(self.bsz, self.len, self.dim)

    def _dummy_sample(self):
        sample = {
            "net_input": {
                "src_tokens": torch.randint(self.voc_size, (self.bsz, self.len)).long()
            }
        }
        return sample

    def _test_direction(self, adversary, name="direction"):
        sample = self._dummy_sample()
        grad = self._dummy_gradient()
        adv_tokens = adversary(sample, grad)
        prev_embeds = self.embed_tokens(sample["net_input"]["src_tokens"])
        new_embeds = self.embed_tokens(adv_tokens)
        delta = new_embeds - prev_embeds
        delta_dot_grad = (delta * grad).sum(-1).sum(-1).data.numpy()
        for b, ddd in enumerate(delta_dot_grad):
            self.assertGreaterEqual(ddd, 0, f"[{name}]: Direction error in batch {b}")

    def _test_decrease_loss(self, adversary, name="decrease_loss"):
        sample = self._dummy_sample()
        src_embeds = self.embed_tokens(sample["net_input"]["src_tokens"])
        src_embeds.retain_grad()
        # Create loss function (simple quadratic loss)
        weights = torch.rand_like(src_embeds)

        def loss_function(x):
            return (weights * (x**2)).sum()
        old_loss = loss_function(src_embeds)
        # Get gradient
        old_loss.backward()
        input_grad = src_embeds.grad
        # Run adversary
        adv_tokens = adversary(sample, input_grad)
        # Compute new loss
        new_embeds = self.embed_tokens(adv_tokens)
        # We only do a small perturbation so that the linear approximation is valid
        unit_delta = F.normalize(new_embeds - src_embeds)
        new_embeds_small_perturb = src_embeds + self.eta * unit_delta
        new_loss = loss_function(new_embeds_small_perturb)
        # Assert
        self.assertLess(new_loss, old_loss, f"[{name}]: Loss did not decrease")

    def test_brute_force_direction(self):
        for max_swaps in range(1, self.len + 1):
            for normalize_directions in [False, True]:
                for _ in range(self.num_random_retries):
                    self._test_direction(
                        self._get_brute_force_adversary(
                            max_swaps, normalize_directions
                        ),
                        name=f"brute_force_direction_{max_swaps}_"
                        f"{normalize_directions}",
                    )

    def test_brute_force_decrease_loss(self):
        for max_swaps in range(1, self.len + 1):
            for normalize_directions in [False, True]:
                for _ in range(self.num_random_retries):
                    self._test_decrease_loss(
                        self._get_brute_force_adversary(
                            max_swaps, normalize_directions
                        ),
                        name=f"brute_force_decrease_loss_{max_swaps}_"
                        f"{normalize_directions}",
                    )
