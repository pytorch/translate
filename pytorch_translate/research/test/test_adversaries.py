#!/usr/bin/env python3

import time
import unittest
from itertools import product
from typing import NamedTuple

import torch
import torch.nn.functional as F
from pytorch_translate.dictionary import Dictionary
from pytorch_translate.research.adversarial import adversaries
from pytorch_translate.tasks import PytorchTranslateTask


# For determinism
torch.manual_seed(11051993)


class ToyEncoder(NamedTuple):
    embed_tokens: torch.nn.Embedding


class ToyModel(NamedTuple):
    encoder: ToyEncoder


class DummyArgs(NamedTuple):
    max_swaps: int
    normalize_directions: bool
    nearest_neighbors: int
    cosine_nn: bool
    alternatives_file: str
    by_gradient_norm: bool
    temperature: float
    forbidden_tokens: list
    allowed_tokens: list
    allow_identity: bool

    def __str__(self):
        return (
            f"{self.max_swaps}_"
            f"{self.normalize_directions}_"
            f"{self.nearest_neighbors}_"
            f"{self.cosine_nn}_"
            f"{self.alternatives_file}_"
            f"{self.by_gradient_norm}_"
            f"{self.temperature}_"
            f"{'&'.join(self.forbidden_tokens)}_"
            f"{'&'.join(self.allowed_tokens)}_"
            f"{self.allow_identity}"
        )


class AdversariesTest(unittest.TestCase):
    def setUp(self):
        self.num_random_retries = 10
        self.n_symbols = 20
        self.dim = 4
        self.bsz = 3
        self.len = 6
        self.eta = 0.001
        # Dictionaries
        self._init_task()
        self.voc_size = len(self.task.src_dict)
        # Model
        self.model = self._toy_model()
        self.embed_tokens = self.model.encoder.embed_tokens
        # Alternatives file
        self._prepare_dummy_alternatives_file()
        # Ranges for grid testing
        self.max_swaps_range = [1, 2, 3]
        self.normalize_directions_range = [False, True]
        self.nearest_neighbors_range = [0, self.voc_size - 1, self.voc_size - 2]
        self.cosine_nn_range = [False, True]
        self.alternatives_file_range = ["", self.alts_file]
        self.by_gradient_norm_range = [False, True]
        self.temperature_range = [0, 1, 10]
        self.forbidden_tokens_range = [[], self.symbols[:1]]
        self.allowed_tokens_range = [[], self.symbols[2:]]
        self.allow_identity_range = [True, False]

    def _toy_model(self):
        embed_tokens = torch.nn.Embedding(self.voc_size, self.dim)
        embed_tokens.weight.data = torch.normal(torch.zeros(self.voc_size, self.dim))
        model = ToyModel(encoder=ToyEncoder(embed_tokens=embed_tokens))
        return model

    def _init_task(self):
        src_dict = Dictionary()
        self.symbols = "abcdefghijklmnopqrstuvwxyz"
        for symbol in self.symbols[: self.n_symbols]:
            src_dict.add_symbol(symbol)
        dst_dict = Dictionary()
        self.task = PytorchTranslateTask(None, src_dict, dst_dict)

    def _get_brute_force_adversary(self, args):
        adv = adversaries.brute_force_adversary.BruteForceAdversary(
            args, self.model, self.task
        )
        return adv

    def _get_random_swap_adversary(self, args):
        adv = adversaries.random_swap_adversary.RandomSwapAdversary(
            args, self.model, self.task
        )
        return adv

    def _dummy_gradient(self):
        return torch.rand(self.bsz, self.len, self.dim)

    def _dummy_sample(self):
        sample = {
            "net_input": {
                "src_tokens": torch.randint(
                    low=self.task.src_dict.nspecial,
                    high=len(self.task.src_dict),
                    size=(self.bsz, self.len),
                ).long()
            }
        }
        return sample

    def _prepare_dummy_alternatives_file(self):
        unique_id = int(time.time())
        self.alts_file = f"/tmp/alts_file_{unique_id}.txt"
        with open(self.alts_file, "w") as af:
            for s_id, symbol in enumerate(self.symbols[: self.n_symbols]):
                # Generate alternatives with basic arithmetic
                # (all symbols except this symbol's neighbors in the list)
                alternatives = [
                    self.symbols[(s_id + idx) % self.n_symbols]
                    for idx in range(2, self.n_symbols - 1)
                ]
                print(" ".join([symbol] + alternatives), file=af)

    def _test_direction(self, adversary, name="direction"):
        """Check that the replacement results in a movement
        in word embedding space that is opposite to the gradient"""
        sample = self._dummy_sample()
        grad = self._dummy_gradient()
        adv_tokens = adversary(sample, grad)
        prev_embeds = self.embed_tokens(sample["net_input"]["src_tokens"])
        new_embeds = self.embed_tokens(adv_tokens)
        delta = new_embeds - prev_embeds
        delta_dot_grad = (delta * grad).sum(-1).sum(-1).data.numpy()
        for b, ddd in enumerate(delta_dot_grad):
            self.assertLessEqual(ddd, 0, f"[{name}]: Direction error in batch {b}")

    def _test_decrease_loss(self, adversary, name="decrease_loss"):
        sample = self._dummy_sample()
        src_embeds = self.embed_tokens(sample["net_input"]["src_tokens"])
        src_embeds.retain_grad()
        # Create loss function (simple quadratic loss)
        weights = torch.rand_like(src_embeds)

        def loss_function(x):
            return (weights * (x ** 2)).sum()

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

    def _test_works(self, adversary, name="works"):
        """Just runs the adversary to verify that it works without bugs"""
        sample = self._dummy_sample()
        grad = self._dummy_gradient()
        adv_tokens = adversary(sample, grad)
        self.embed_tokens(adv_tokens)
        # Check that the new tokens are not special tokens (except for UNK)
        for tok in adv_tokens.view(-1):
            self.assertTrue(
                tok == self.task.src_dict.unk_index
                or tok >= self.task.src_dict.nspecial,
                f"[{name}]: forbidden token {self.task.src_dict[tok]}",
            )

    def _brute_force_grid_args(self):
        """Generate all combinations of arguments we're interested in testing
        for the brute force adversary"""
        grid_args = product(
            self.max_swaps_range,
            self.normalize_directions_range,
            self.nearest_neighbors_range,
            self.cosine_nn_range,
            self.alternatives_file_range,
            [False],  # by_gradient_norm
            [0],  # temperature
            self.forbidden_tokens_range,
            self.allowed_tokens_range,
            self.allow_identity_range,
        )
        for args in grid_args:
            nearest_neighbors = args[2]
            cosine_nn = args[3]
            # Ignore this combination
            if nearest_neighbors == 0 and cosine_nn:
                continue
            else:
                yield DummyArgs(*args)

    def _random_swap_grid_args(self):
        """Generate all combinations of arguments we're interested in testing
        for the random adversary"""
        grid_args = product(
            self.max_swaps_range,
            [False],
            self.nearest_neighbors_range,
            self.cosine_nn_range,
            self.alternatives_file_range,
            self.by_gradient_norm_range,
            self.temperature_range,
            self.forbidden_tokens_range,
            self.allowed_tokens_range,
            self.allow_identity_range,
        )
        for args in grid_args:
            nearest_neighbors = args[2]
            cosine_nn = args[3]
            by_gradient_norm = args[5]
            temperature = args[6]
            # Ignore those combinations
            if nearest_neighbors == 0 and cosine_nn:
                continue
            elif not by_gradient_norm and temperature != 1:
                continue
            else:
                yield DummyArgs(*args)

    def test_brute_force_direction(self):
        for args in self._brute_force_grid_args():
            print(f"Testing {args}", flush=True)
            for _ in range(self.num_random_retries):
                self._test_direction(
                    self._get_brute_force_adversary(args),
                    name=f"brute_force_direction_{args}",
                )

    def test_brute_force_decrease_loss(self):
        # Only test for loss decrease with the default constraints
        args = next(self._brute_force_grid_args())
        print(f"Testing {args}", flush=True)
        for n_try in range(self.num_random_retries):
            self._test_decrease_loss(
                self._get_brute_force_adversary(args),
                name=f"brute_force_direction_{args}_#{n_try}",
            )

    def test_brute_force_works(self):
        for args in self._brute_force_grid_args():
            print(f"Testing {args}", flush=True)
            for n_try in range(self.num_random_retries):
                self._test_works(
                    self._get_brute_force_adversary(args),
                    name=f"brute_force_works_{args}_#{n_try}",
                )

    def test_random_swap_works(self):
        for args in self._random_swap_grid_args():
            print(f"Testing {args}", flush=True)
            for n_try in range(self.num_random_retries):
                self._test_works(
                    self._get_random_swap_adversary(args),
                    name=f"random_swap_works_{args}_#{n_try}",
                )
