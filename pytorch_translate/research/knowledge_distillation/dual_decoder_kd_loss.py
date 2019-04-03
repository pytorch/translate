#!/usr/bin/env python3

import math

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("dual_decoder_kd_loss")
class DualDecoderCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.kd_weight = args.kd_weight

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="[teacher decoder only] epsilon for label smoothing, 0 means "
            "no label smoothing.",
        )
        parser.add_argument(
            "--kd-weight",
            type=float,
            default=0.5,
            help=(
                "[student decoder only] mixture weight between the knowledge "
                "distillation and negative log likelihood losses. Must be in "
                "[0.0, 1.0].",
            ),
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]

        encoder_out = model.encoder(src_tokens, src_lengths)
        student_output = model.student_decoder(prev_output_tokens, encoder_out)
        teacher_output = model.teacher_decoder(prev_output_tokens, encoder_out)

        teacher_loss, teacher_nll_loss, teacher_probs = self.compute_teacher_loss(
            model, teacher_output, sample, reduce=reduce
        )

        # do not propagate gradient from student loss to teacher output
        teacher_probs = teacher_probs.detach()
        student_loss, student_nll_loss = self.compute_student_loss(
            model, student_output, sample, teacher_probs, reduce=reduce
        )

        total_loss = student_loss + teacher_loss

        sample_size = (
            sample["target"].size(0) if self.args.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "teacher_loss": utils.item(teacher_loss.data)
            if reduce
            else teacher_loss.data,
            "teacher_nll_loss": utils.item(teacher_nll_loss.data)
            if reduce
            else teacher_nll_loss.data,
            "student_loss": utils.item(student_loss.data)
            if reduce
            else student_loss.data,
            "student_nll_loss": utils.item(student_nll_loss.data)
            if reduce
            else student_nll_loss.data,
            "loss": utils.item(total_loss.data) if reduce else total_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return total_loss, sample_size, logging_output

    def compute_teacher_loss(self, model, net_output, sample, reduce=True):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        probs = probs.view(-1, probs.size(-1))
        lprobs = probs.log()
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1.0 - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss, probs

    def compute_student_loss(
        self, model, net_output, sample, teacher_probs, reduce=True
    ):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        kd_loss = (teacher_probs * -lprobs).sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            kd_loss = kd_loss.sum()
        loss = (1.0 - self.kd_weight) * nll_loss + self.kd_weight * kd_loss
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        return {
            "student_loss": sum(log.get("student_loss", 0) for log in logging_outputs)
            / sample_size
            / math.log(2),
            "student_nll_loss": sum(
                log.get("student_nll_loss", 0) for log in logging_outputs
            )
            / ntokens
            / math.log(2),
            "teacher_loss": sum(log.get("teacher_loss", 0) for log in logging_outputs)
            / sample_size
            / math.log(2),
            "teacher_nll_loss": sum(
                log.get("teacher_nll_loss", 0) for log in logging_outputs
            )
            / ntokens
            / math.log(2),
            "loss": sum(log.get("loss", 0) for log in logging_outputs)
            / sample_size
            / math.log(2),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
