#!/usr/bin/env python3

from fairseq import utils
from fairseq.criterions import LegacyFairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
)


@register_criterion("weighted_label_smoothed_cross_entropy")
class WeightedLabelSmoothedCrossEntropyCriterion(LegacyFairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @classmethod
    def add_args(cls, parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        assert "weights" in sample, "Need to specify weights for examples."
        weights = sample["weights"].unsqueeze(1).unsqueeze(2)
        lprobs = lprobs * weights

        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1.0 - self.eps) * nll_loss + eps_i * smooth_loss

        sample_size = (
            sample["target"].size(0) if self.args.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @classmethod
    def aggregate_logging_outputs(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        return LabelSmoothedCrossEntropyCriterion.aggregate_logging_outputs(
            logging_outputs
        )
