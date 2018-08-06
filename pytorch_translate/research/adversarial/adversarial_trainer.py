#!/usr/bin/env python3

from collections import defaultdict, OrderedDict
import torch

from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, TimeMeter


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
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()       # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['oom'] = AverageMeter()    # out of memory
        self.meters['wall'] = TimeMeter()      # wall time in seconds

    def gen_adversarial_examples(self, sample):
        """Get adversarial examples from existing sample"""

        sample = self._prepare_sample(sample)

        # Set the model to adversarial mode
        self.model.encoder.set_gradient_tracking_mode(True)

        # forward pass
        (
            loss, sample_sizes, logging_outputs, ooms_fwd
        ) = self._forward_adversarial(sample)

        # aggregate stats and logging outputs
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)

        # backward pass, get input_gradients
        input_gradients, ooms_bwd = self._get_gradients_wrt_input(loss)

        # perturbate input
        adversarial_input = self.adversary(sample, input_gradients)

        # update meters
        self.meters["wps"].update(ntokens)
        self.meters["wpb"].update(ntokens)
        self.meters["bsz"].update(nsentences)
        self.meters["oom"].update(ooms_fwd + ooms_bwd)

        # Deactivate adversarial mode
        self.model.encoder.set_gradient_tracking_mode(False)

        return adversarial_input

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
                (
                    loss, sample_size, logging_output_
                ) = self.adversarial_criterion(self.model, sample)
                logging_output.update(logging_output_)
            except RuntimeError as e:
                if not eval and "out of memory" in str(e):
                    print("| WARNING: ran out of memory, skipping batch")
                    oom = 1
                    loss = None
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                else:
                    raise e

        # synchronize logging outputs for multi-GPU training
        if getattr(self.args, "distributed_world_size", 1) > 1:
            raise ValueError("Distributed adversarial example generations is "
            "not supported yet")
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
        if loss is not None:
            try:
                # backward pass
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("| WARNING: ran out of memory, skipping batch")
                    oom = 1
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                else:
                    raise e

        # all-reduce grads and rescale by grad_denom
        if getattr(self.args, "distributed_world_size", 1) > 1:
            raise ValueError("Distributed adversarial example generations is "
            "not supported yet")
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
