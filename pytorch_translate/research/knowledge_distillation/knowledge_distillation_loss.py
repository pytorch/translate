#!/usr/bin/env python3

import math
import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from pytorch_translate import utils as pytorch_translate_utils


@register_criterion('word_knowledge_distillation')
class KnowledgeDistillationCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        assert args.teacher_path, (
            'Please specify at least one valid file for --teacher-path'
        )
        use_cuda = torch.cuda.is_available() and not self.args.cpu

        # Load model ensemble from checkpoints
        self.teacher_models, self.teacher_model_args = (
            pytorch_translate_utils.load_diverse_ensemble_for_inference(
                [args.teacher_path], task,
            )
        )

        # Move models to device and to evaluation mode
        if use_cuda:
            for model in self.teacher_models:
                model.cuda()
        for model in self.teacher_models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam
            )

        self.kd_weight = getattr(args, 'kd_weight', 0)
        if self.kd_weight < 0 or self.kd_weight > 1:
            raise ValueError(
                f'--kd-weight ({self.kd_weight}) must be in [0, 1]'
            )

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            '--teacher-path',
            metavar='FILE',
            action='append',
            help='path(s) to teacher model file(s)',
        )
        parser.add_argument(
            '--kd-weight',
            type=float,
            default=0.0,
            help=(
                'mixture weight between the knowledge distillation and',
                'negative log likelihood losses. Must be in [0.0, 1.0]',
            )
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # 1. Generate translation using student model
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # [bsz, seqlen, vocab] -> [bsz*seqlen, vocab]
        lprobs = lprobs.view(-1, lprobs.size(-1))

        # 2. Generate translation using teacher models
        avg_probs = None
        for teacher_model in self.teacher_models:
            teacher_output = teacher_model(**sample['net_input'])
            probs = teacher_model.get_normalized_probs(
                teacher_output,
                log_probs=False,
            )
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
        avg_probs.div_(len(self.teacher_models))
        avg_probs = avg_probs.view(-1, avg_probs.size(-1)).detach()
        kd_loss = -torch.sum(avg_probs * lprobs)

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
        loss = kd_loss * self.kd_weight + nll_loss * (1 - self.kd_weight)

        if self.args.sentence_avg:
            sample_size = sample['target'].size(0)
        else:
            sample_size = sample['ntokens']
        if self.args.sentence_avg:
            sample_size = sample['target'].size(0)
        else:
            sample_size = sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
