# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from pytorch_translate.rescoring.model_scorers import SimpleModelScorer
from pytorch_translate.rescoring.rescorer import (  # noqa
    Rescorer,
    add_args as rescore_add_args,
    combine_weighted_scores,
)
from pytorch_translate.research.rescore.rescoring_criterion import RescoringCriterion


@register_criterion("rescoring_word_criterion")
class RescoringWordCriterion(RescoringCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        RescoringCriterion.add_args(parser)
        parser.add_argument(
            "--topk-words",
            type=int,
            default=8,
            help="match topk words at each time step",
        )
        parser.add_argument(
            "--word-weight", type=float, default=1.0, help="weight for word level"
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_tokens = sample["net_input"]["src_tokens"]
        bsz, srclen = src_tokens.size()
        encoder_input = {
            k: v
            for k, v in sample["net_input"].items()
            if k in ["src_tokens", "src_lengths"]
        }

        # 1) Generate hypos
        hypos = [[] for j in range(bsz)]
        for k in range(bsz):
            hypos[k] = [{"tokens": sample["target"][k]}]
        hypos = [hypo for _ in hypos for hypo in _]

        # 2) Compute (log)-probs via forward models
        self.self_rescorer.model = model
        self.self_rescorer.task = self.task
        model.train()
        assert self.self_rescorer.model.training, "model should be in training phase"

        hypo_encoder_inputs, hypo_tokens = self.self_rescorer.prepare_inputs(
            src_tokens, hypos
        )
        hypo_logprobs, hypo_encoder_outs, forward_logprobs = self.self_rescorer.score_tokens(
            hypo_encoder_inputs, hypo_tokens
        )

        # 3) Compute rewards from rescoring left-right models
        with torch.no_grad():
            teacher_model = self.rescore_models["cloze_model"]
            rescorer = SimpleModelScorer(self.args, None, teacher_model, self.task)
            _, _, rescore_logprobs = rescorer.score_tokens(
                hypo_encoder_inputs, hypo_tokens
            )

        # 4) Word-level loss
        f_logprob, f_index = forward_logprobs.topk(self.args.topk_words)

        ## KL(p_s || p_t) = \sum p_s log p_s - \sum p_s log p_t, aka RL + maxEnt
        word_loss = (
            f_logprob.exp() * (f_logprob - 1.0 * rescore_logprobs.gather(-1, f_index))
        ).sum()

        # 5) Compute Cross Entropy loss
        eos = self.task.target_dictionary.eos()
        target_tokens = torch.cat(
            (
                torch.zeros(bsz, 1).fill_(eos).type_as(sample["target"]),
                sample["target"],
            ),
            dim=1,
        )
        target_encoder_inputs = (
            encoder_input["src_tokens"],
            [encoder_input["src_lengths"][0].item()],
        )
        target_logprobs, target_encoder_out, _ = self.self_rescorer.score_tokens(
            target_encoder_inputs, target_tokens
        )

        # 6) Gather losses
        loss = self.args.word_weight * word_loss - target_logprobs.sum()

        # Logging
        sample_size = (
            sample["target"].size(0) if self.args.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output
