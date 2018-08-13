#!/usr/bin/env python3

import math

import numpy as np
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import CRITERION_REGISTRY, FairseqCriterion, register_criterion


def build_criterion(args, task):
    """Same as fairseq.criterions.build_criterion but for adversarial criterion"""
    return CRITERION_REGISTRY[args.adv_criterion](args, task)


@register_criterion("all_bad_words")
class AllBadWordsCriterion(FairseqCriterion):
    """This is essentially the negation of CrossEntropyCriterion.

    Instead of optimizing P[w1 AND w2 AND...] we optimize
    P[(NOT w1) AND (NOT w2) AND...]
    Notice that this is *not* the same as reversing the nll objective (which
    would mean optimizing P[(NOT w1) OR (NOT w2) OR...])
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # 1. Generate log probabilities according to the model
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # B x T x |V| -> (B*T) x |V|
        lprobs = lprobs.view(-1, lprobs.size(-1))

        # 2. Compute log-probability of not producing any valid word
        target = model.get_targets(sample, net_output).view(-1)
        lp_targets = lprobs.index_select(dim=1, index=target)
        # Negation in the log-semiring (is that stable?)
        lp_not_targets = torch.log(1 - torch.exp(lp_targets))
        # Masking
        mask_pads = target.eq(self.padding_idx)
        lp_not_targets = lp_not_targets.masked_fill(mask_pads, 0)
        # Sum everything (=AND)
        if reduce:
            loss = lp_not_targets.sum()
        else:
            loss = lp_not_targets.view(net_output.size(0), -1).sum(-1)

        # Negate because we're minimizing the criterion
        loss = -loss

        if self.args.sentence_avg:
            sample_size = sample["target"].size(0)
        else:
            sample_size = sample["ntokens"]

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size / math.log(2),
            "sample_size": sample_size,
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)
        return agg_output


@register_criterion("force_words")
class ForceWordsCriterion(FairseqCriterion):
    """This forces the model to generate a specific word or set of words."""

    def __init__(self, args, task):
        super().__init__(args, task)
        idxs_list = [task.tgt_dict.index(word) for word in args.words_list]
        self.target_tokens = torch.LongTensor(idxs_list)
        if torch.cuda.is_available():
            self.target_tokens = self.target_tokens.cuda()
        self.eos_idx = task.tgt_dict.eos()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--words-list",
            default=["potato"],
            nargs="+",
            metavar="WORDS",
            help="Words the adversary will try to force the model to predict",
        )
        parser.add_argument(
            "--force-not",
            default=False,
            action="store_true",
            help="Force the model *not* to generate the specified words.",
        )
        parser.add_argument(
            "--only-first",
            default=False,
            action="store_true",
            help="Apply the criterion only to the first word of the output.",
        )
        parser.add_argument(
            "--any-position",
            default=False,
            action="store_true",
            help="This will enforce that the model generates the word at LEAST "
            "once. Think of it as doing OR along the length dimension (as "
            "opposed to AND). This is superseded by --only-first.",
        )
        parser.add_argument(
            "--mask-eos",
            default=False,
            action="store_true",
            help="Exclude the EOS token from the objective.",
        )

    def log_likelihood_loss(self, model, sample, net_output):
        # 1. Generate log probabilities according to the model
        lprobs = model.get_normalized_probs(net_output, log_probs=True)

        # 2. Retrieve the logprobs for each target word
        #    B x T x |V| -> B x T x |target_tokens|
        target_tokens_lprobs = lprobs.index_select(dim=2, index=self.target_tokens)

        # 3. We want to predict the any of the words so we take the sum in the
        #    log semiring (=OR)
        #    B x T x |target_tokens| -> B x T
        any_target_token_lprob = target_tokens_lprobs.logsumexp(2)
        if self.args.force_not:
            any_target_token_lprob = torch.log(1.0 - torch.exp(any_target_token_lprob))
        # Reduce along length and return
        return -self.reduce_along_length(any_target_token_lprob, sample)

    def reduce_along_length(self, loss_by_position, sample):
        # Sum over all positions (=AND)
        # B x T -> B
        if self.args.only_first:
            final_loss = loss_by_position[:, 0]
        else:
            # We will mask the padding tokens
            is_pad_token = sample["target"].eq(self.padding_idx)

            if self.args.mask_eos:
                # Mask the EOS token as well
                is_eos_token = sample["target"].eq(self.eos_idx)
                is_pad_token = is_pad_token | is_eos_token

            # Depending on the loss and the reduction approach we have different
            # masking values
            if self.args.any_position:
                # For the logsumexp the mask value is -infinity
                masked_loss = loss_by_position.masked_fill(is_pad_token, -np.inf)
                final_loss = masked_loss.logsumexp(1)
            else:
                # By default we take the sum so the mask value is 0
                masked_loss = loss_by_position.masked_fill(is_pad_token, 0)
                final_loss = masked_loss.sum(1)

        return final_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])

        final_loss = self.log_likelihood_loss(model, sample, net_output)

        # Sum over batch elements
        if reduce:
            # B -> 1
            final_loss = final_loss.sum()

        if getattr(self.args, "sentence_avg", False):
            sample_size = sample["target"].size(0)
        else:
            sample_size = sample["ntokens"]

        logging_output = {
            "loss": utils.item(final_loss.data) if reduce else final_loss.data,
            "ntokens": sample["ntokens"],
            "sample_size": sample_size,
        }
        return final_loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size / math.log(2),
            "sample_size": sample_size,
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)
        return agg_output


@register_criterion("force_words_hinge")
class ForceWordsHingeCriterion(FairseqCriterion):
    """This forces the model to generate a specific word or set of words."""

    def __init__(self, args, task):
        super().__init__(args, task)
        idxs_list = [task.tgt_dict.index(word) for word in args.words_list]
        self.target_tokens = torch.LongTensor(idxs_list)
        if torch.cuda.is_available():
            self.target_tokens = self.target_tokens.cuda()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--words-list",
            default=["potato"],
            nargs="+",
            metavar="WORDS",
            help="Words the adversary will try to force the model to predict",
        )
        parser.add_argument(
            "--force-not",
            default=False,
            action="store_true",
            help="Force the model *not* to generate the specified words.",
        )
        parser.add_argument(
            "--only-first",
            default=False,
            action="store_true",
            help="Apply the criterion only to the first word of the output.",
        )
        parser.add_argument(
            "--hinge-slack",
            default=1.0,
            type=float,
            help="Slack variable for the hinge loss.",
        )
        parser.add_argument(
            "--any-position",
            default=False,
            action="store_true",
            help="This will enforce that the model generates the word at LEAST "
            "once. Think of it as doing OR along the length dimension (as "
            "opposed to AND). This is superseded by --only-first.",
        )

        parser.add_argument(
            "--topk",
            default=1,
            type=int,
            help="This will sum the max margin objective over the top-k logits. "
            "Maybe useful in case we want to break beam search.",
        )

    def hinge_loss(self, sample, net_output):
        # 1. Retrieve logits
        logits = net_output[0]

        # 2. Pick up the llogits of the target words
        target_tokens_logits = logits.index_select(dim=2, index=self.target_tokens)

        # 3. Take the max over the other tokens
        logits[:, :, self.target_tokens] = -np.inf
        max_logits, _ = logits.topk(self.args.topk, dim=2)

        # 4. Compute the hinge loss
        margin = max_logits.unsqueeze(-1) - target_tokens_logits.unsqueeze(2)
        if self.args.force_not:
            # Reverse the margin if we want to force the model not to generate
            # the words
            margin = -margin
        # Compute the hinge loss and reduce (here min works better than sum
        # from my experience)
        hinge_loss = F.relu(margin + self.args.hinge_slack).sum(2)

        # 5. At each timestep choose the easiest option (this is important)
        loss_at_position, _ = hinge_loss.min(2)

        # 6. Then reduce along the length dimension first
        reduced_loss = self.reduce_along_length(loss_at_position, sample)

        return reduced_loss

    def reduce_along_length(self, loss_by_position, sample):
        # Sum over all positions (=AND)
        # B x T -> B
        if self.args.only_first:
            final_loss = loss_by_position[:, 0]
        else:
            # We will mask the padding tokens
            is_pad_token = sample["target"].eq(self.padding_idx)

            # Depending on the reduction approach we have different mask values
            if self.args.any_position:
                # For the hinge loss we take the min so the mask value is +infinity
                masked_loss = loss_by_position.masked_fill(is_pad_token, np.inf)
                final_loss, _ = masked_loss.min(1)
            else:
                # By default we take the sum so the mask value is 0
                masked_loss = loss_by_position.masked_fill(is_pad_token, 0)
                final_loss = masked_loss.sum(1)

        return final_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])

        final_loss = self.hinge_loss(sample, net_output)

        # Sum over batch elements
        if reduce:
            # B -> 1
            final_loss = final_loss.sum()

        if getattr(self.args, "sentence_avg", False):
            sample_size = sample["target"].size(0)
        else:
            sample_size = sample["ntokens"]

        logging_output = {
            "loss": utils.item(final_loss.data) if reduce else final_loss.data,
            "ntokens": sample["ntokens"],
            "sample_size": sample_size,
        }
        return final_loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size / math.log(2),
            "sample_size": sample_size,
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)
        return agg_output
