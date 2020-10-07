#!/usr/bin/env python3

import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import LegacyFairseqCriterion, register_criterion
from pytorch_translate import utils as pytorch_translate_utils


@register_criterion("word_knowledge_distillation")
class KnowledgeDistillationCriterion(LegacyFairseqCriterion):
    def __init__(self, args, task):
        """
        This code is for word-level knowledge distillation. Most of the algorithm
        is inspired from the Kim and Rush (2016) paper:
        http://www.aclweb.org/anthology/D16-1139
        """
        super().__init__(args, task)
        self.kd_weight = getattr(args, "kd_weight", 0)
        if self.kd_weight < 0 or self.kd_weight > 1:
            raise ValueError(f"--kd-weight ({self.kd_weight}) must be in [0, 1]")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--kd-weight",
            type=float,
            default=0.0,
            help=(
                "mixture weight between the knowledge distillation and",
                "negative log likelihood losses. Must be in [0.0, 1.0]",
            ),
        )

    def get_kd_loss(self, sample, student_lprobs, lprobs):
        """
        The second return argument is used for unit testing.

        Args:
            * sample: batched sample that has teacher score keys (top_k_scores and
             top_k_indices)
            * student_lprobs: tensor of student log probabilities
            * lprobs: flat version of student_lprobs
        """
        top_k_teacher_probs_normalized = sample["top_k_scores"]
        indices = sample["top_k_indices"]

        assert indices.shape[0:1] == student_lprobs.shape[0:1]

        kd_loss = -(
            torch.sum(
                torch.gather(student_lprobs, 2, indices)
                * top_k_teacher_probs_normalized.float()
            )
        )
        return kd_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # 1. Generate translation using student model
        net_output = model(**sample["net_input"])
        student_lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # [bsz, seqlen, vocab] -> [bsz*seqlen, vocab]
        lprobs = student_lprobs.view(-1, student_lprobs.size(-1))

        # 2. Get translation from teacher models and calulate KD loss.
        kd_loss = None
        if "top_k_scores" in sample:
            # top_k_scores is not present in the validation data.
            kd_loss = self.get_kd_loss(sample, student_lprobs, lprobs)

        # 3. Compute NLL loss with respect to the ground truth
        target = model.get_targets(sample, net_output).view(-1)
        nll_loss = F.nll_loss(
            lprobs,
            target,
            size_average=False,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        # 4. Linearly interpolate between NLL and KD loss
        if kd_loss is not None:
            loss = kd_loss * self.kd_weight + nll_loss * (1 - self.kd_weight)
        else:
            loss = nll_loss

        if self.args.sentence_avg:
            sample_size = sample["target"].size(0)
        else:
            sample_size = sample["ntokens"]
        if self.args.sentence_avg:
            sample_size = sample["target"].size(0)
        else:
            sample_size = sample["ntokens"]
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            "nsamples": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size / math.log(2),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)
        return agg_output
