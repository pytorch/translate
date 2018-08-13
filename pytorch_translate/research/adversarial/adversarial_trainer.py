#!/usr/bin/env python3

from collections import OrderedDict, defaultdict
from itertools import chain

import torch
import torch.nn.functional as F
from fairseq import distributed_utils
from fairseq.meters import AverageMeter, TimeMeter
from fairseq.trainer import Trainer

from .adversarial_utils import clone_sample, detach_sample, tile


class AdversarialTrainer(Trainer):
    """This is a modified trainer to handle adversarial examples generation

    It takes in two additional arguments:

    - `adversarial_criterion`: The objective to minimize for adversarial
       examples (eg. the log-likelihood)
    - `adversary`: An `Adversary` object that will generate and adversarial
       input given the original input and the gradients of the adversarial
       criterion wrt. those inputs

    Having adversarial attacks part of the trainer makes it possible to generate
    adversarial examples on the fly during training (this is still TODO).
    """

    def __init__(self, args, task, model, criterion, adversarial_criterion, adversary):
        # We don't inherit the constructor to handle the case when there is no
        # optimizer/lr_scheduler, ie when we're only interested in the
        # adversarial generation aspect
        if not torch.cuda.is_available():
            raise NotImplementedError("Training on CPU is not supported.")

        self.args = args

        # copy model and criterion to current device
        # TODO: is there an easy way to make this available for CPU. And if so,
        #       why is training on CPU not supported by fairseq?
        self.task = task
        self.model = model.cuda()
        self.criterion = criterion.cuda() if criterion is not None else None
        self.adversarial_criterion = adversarial_criterion.cuda()
        self.adversary = adversary.cuda()

        # Weighting
        self.adv_weight = getattr(args, "adv_weight", 0)

        # initialize meters
        self.init_meters()

        self._max_bsz_seen = 0
        self._num_updates = 0

        self.reverse_criterion = args.reverse_criterion

        self._buffered_stats = defaultdict(lambda: [])
        self._flat_grads = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None

    def init_meters(self):
        """Initialize the meters for logging"""
        self.meters = OrderedDict()
        self.meters["train_loss"] = AverageMeter()
        self.meters["train_nll_loss"] = AverageMeter()
        self.meters["valid_loss"] = AverageMeter()
        self.meters["valid_nll_loss"] = AverageMeter()
        self.meters["wps"] = TimeMeter()  # words per second
        self.meters["ups"] = TimeMeter()  # updates per second
        self.meters["wpb"] = AverageMeter()  # words per batch
        self.meters["bsz"] = AverageMeter()  # sentences per batch
        self.meters["gnorm"] = AverageMeter()  # gradient norm
        self.meters["clip"] = AverageMeter()  # % of updates clipped
        self.meters["oom"] = AverageMeter()  # out of memory
        self.meters["wall"] = TimeMeter()  # wall time in seconds

    def gen_adversarial_examples(self, sample):
        """Get adversarial examples from existing sample"""

        sample = self._prepare_sample(sample)

        # Set the model to adversarial mode
        self.model.encoder.set_gradient_tracking_mode(True)

        # Keep track of ooms
        ooms_adv = 0

        # The initial adversarial input is just the original input
        adversarial_input = sample["net_input"]["src_tokens"]

        for _ in range(self.args.n_attack_iterations):
            # Set sample input tokens to the adversarial input
            sample["net_input"]["src_tokens"] = adversarial_input.detach()

            # forward pass
            (loss, sample_sizes, logging_outputs, ooms_fwd) = self._forward_adversarial(
                sample
            )

            # aggregate stats and logging outputs
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)

            # backward pass, get input_gradients
            input_gradients, ooms_bwd = self._get_gradients_wrt_input(loss)

            # Get adversarial inputs from the adversary
            adversarial_input, ooms_attack = self._get_adv_input(
                sample, input_gradients
            )

            # Zero gradients so that gradients from adversarial example
            # generation don't pollute training
            self.model.zero_grad()

            # Update ooms at each iteration
            ooms_adv += ooms_fwd + ooms_bwd + ooms_attack
            self.meters["oom"].update(ooms_adv)
            # Abort if there is an OOM
            if ooms_adv > 0:
                break

        # update other meters after all iterations
        self.meters["wps"].update(ntokens)
        self.meters["wpb"].update(ntokens)
        self.meters["bsz"].update(nsentences)

        # Deactivate adversarial mode
        self.model.encoder.set_gradient_tracking_mode(False)

        return adversarial_input.detach(), ooms_adv

    def _get_adv_input(self, sample, input_gradients):
        """A wrapper around the call to the adversary robust to OOMs"""
        oom = 0
        adversarial_input = sample["net_input"]["src_tokens"].detach()
        if input_gradients is not None:
            try:
                # Modify gradients if needs be
                if self.args.modify_gradient == "sign":
                    input_gradients = torch.sign(input_gradients)
                elif self.args.modify_gradient == "normalize":
                    input_gradients = F.normalize(input_gradients, dim=2)

                # perturbate input
                adversarial_input = self.adversary(sample, input_gradients)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        "| WARNING: ran out of memory in the attack phase of "
                        "an adversarial attack, skipping batch"
                    )
                    oom = 1
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                else:
                    raise e

        return adversarial_input, oom

    def _forward_adversarial(self, sample):
        # Set model to training mode
        self.model.train()
        # But disable dropout
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()

        loss = None
        sample_size = 0
        logging_output = {
            "ntokens": sample["ntokens"] if sample is not None else 0,
            "nsentences": sample["target"].size(0) if sample is not None else 0,
        }
        oom = 0
        if sample is not None:
            try:
                # calculate loss and sample size
                (loss, sample_size, logging_output_) = self.adversarial_criterion(
                    self.model, sample
                )
                logging_output.update(logging_output_)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        "| WARNING: ran out of memory in the forward pass of "
                        "an adversarial attack, skipping batch"
                    )
                    oom = 1
                    loss = None
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                else:
                    raise e

        # synchronize logging outputs for multi-GPU training
        if getattr(self.args, "distributed_world_size", 1) > 1:
            raise ValueError(
                "Distributed adversarial example generations is " "not supported yet"
            )
        else:
            sample_sizes = [sample_size]
            logging_outputs = [logging_output]
            ooms = oom

        # Negate the loss in order to maximize it
        if self.reverse_criterion:
            loss = -loss

        return loss, sample_sizes, logging_outputs, ooms

    def _get_gradients_wrt_input(self, loss):
        """Gets the gradient of a loss wrt the token embeddings"""
        # Safe backward
        oom = 0
        input_gradient = None
        if loss is not None:
            try:
                # backward pass
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        "| WARNING: ran out of memory in the backward pass of "
                        "an adversarial attack, skipping batch"
                    )
                    oom = 1
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                    if self.optimizer is not None:
                        self.optimizer.zero_grad()
                else:
                    raise e

        # all-reduce grads and rescale by grad_denom
        if getattr(self.args, "distributed_world_size", 1) > 1:
            raise ValueError(
                "Distributed adversarial example generations is not supported yet"
            )
        else:
            try:
                token_embeds = self.model.encoder.tracker["token_embeddings"]
            except AttributeError as e:
                raise ValueError(
                    "It looks like the encoder does not support variable "
                    "tracking. Make sure that it has a `tracker` attribute and "
                    "that it keeps track of the token embeddings."
                )

            input_gradient = token_embeds.grad

        return input_gradient, oom

    def train_step(self, sample, update_params=True, augment_adv=False):
        """Do adv sample generation, forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Memory leak without this line
        torch.cuda.empty_cache()

        # Create variables
        sample = self._prepare_sample(sample)

        if augment_adv and self.adv_weight > 0:
            # Get adv input
            adv_input, ooms_adv = self.gen_adversarial_examples(detach_sample(sample))

            if self.args.accumulate_adv_gradient:
                # Perform a forward/backward pass on the adversarial input
                adv_sample = clone_sample(sample)
                adv_sample["net_input"]["src_tokens"] = adv_input
                # Weigh samples accordingly
                sample["weights"] = (1 - self.adv_weight) * torch.ones_like(
                    sample["net_input"]["src_lengths"]
                ).float()
                adv_sample["weights"] = (
                    self.adv_weight
                    * torch.ones_like(adv_sample["net_input"]["src_lengths"]).float()
                )
                # Compute parameter gradients on the adversarial input
                super(AdversarialTrainer, self).train_step(
                    adv_sample, update_params=False
                )
            else:
                # Add the adversarial input to the sample, effectively
                # creating one big batch. This will be faster (because
                # of GPU parallelization) but takes more memory.
                sample, ooms_inc = self._incorporate_adv_input_to_sample(
                    sample, adv_input
                )
                ooms_adv += ooms_inc

            self.meters["oom"].update(ooms_adv)

        agg_logging_output = super(AdversarialTrainer, self).train_step(
            sample, update_params=update_params
        )

        return agg_logging_output

    def _incorporate_adv_input_to_sample(self, sample, adv_input):
        """Interleaves normal input and adv input
        (assuming they have the same length)"""
        oom = 0
        try:
            net_input = sample["net_input"]
            # source tokens
            augmented_src_tokens = tile(net_input["src_tokens"], 0, 2)
            augmented_src_tokens[1::2] = adv_input
            net_input["src_tokens"] = augmented_src_tokens

            # Duplicate the other variables
            net_input["src_lengths"] = tile(net_input["src_lengths"], 0, 2)
            net_input["prev_output_tokens"] = tile(
                net_input["prev_output_tokens"], 0, 2
            )
            # Necessary?
            sample["net_input"] = net_input
            sample["target"] = tile(sample["target"], 0, 2)
            # Weights
            # We want the final loss to be (1-w) * loss_src + w * loss_adv
            # However pytorch will automatically average the loss over the batch,
            # effectively giving 0.5 * ((1-w) * loss_src + w * loss_adv)
            # therefore we multiply each weight by 2 so that the average weight
            # of each batch element is 1
            sample["weights"] = torch.ones_like(net_input["src_lengths"]).float()
            sample["weights"][::2] = (1 - self.adv_weight) * 2
            sample["weights"][1::2] = self.adv_weight * 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    "| WARNING: ran out of memory while creating adversarially "
                    "augmented batch, reverting to single batch."
                )
                oom = 1
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
            else:
                raise e

        return clone_sample(sample), oom
