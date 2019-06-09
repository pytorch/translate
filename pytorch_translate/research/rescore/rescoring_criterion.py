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
from pytorch_translate import generate, utils as pytorch_translate_utils
from pytorch_translate.rescoring.model_scorers import SimpleModelScorer
from pytorch_translate.rescoring.rescorer import (  # noqa
    Rescorer,
    add_args as rescore_add_args,
    combine_weighted_scores,
)


@register_criterion("rescoring_criterion")
class RescoringCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.self_rescorer = SimpleModelScorer(args, None, None, task)
        self.rescore_models = self.load_rescore_models(args)
        self.args = args
        self.task = task

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        rescore_add_args(parser)
        parser.add_argument(
            "--rl-weight",
            type=float,
            default=0.1,
            help="trade-off coefficient of rl loss",
        )
        parser.add_argument(
            "--rl-num-trajectory",
            type=int,
            default=3,
            help="num trajectory in rl training",
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_tokens = sample["net_input"]["src_tokens"]
        beam_size = self.args.rl_num_trajectory
        bsz, srclen = src_tokens.size()
        encoder_input = {
            k: v
            for k, v in sample["net_input"].items()
            if k in ["src_tokens", "src_lengths"]
        }

        # 1) Generate hypos
        translator = generate.build_sequence_generator(self.args, self.task, [model])
        with torch.no_grad():
            hypos = translator.generate(
                encoder_input,
                beam_size,
                maxlen=int(self.args.max_len_a * srclen + self.args.max_len_b),
            )
        hypos = [hypo for _ in hypos for hypo in _]
        hypos_len = torch.tensor(
            [len(hypo["tokens"]) for hypo in hypos]
        ).type_as(src_tokens).float()

        # 2) Compute (log)-probs via forward models
        self.self_rescorer.model = model
        self.self_rescorer.task = self.task
        model.train()
        assert self.self_rescorer.model.training, "model should be in training phase"

        hypo_encoder_inputs, hypo_tokens = self.self_rescorer.prepare_inputs(
            src_tokens, hypos
        )
        hypo_logprobs, hypo_encoder_outs, _ = self.self_rescorer.score_tokens(
            hypo_encoder_inputs, hypo_tokens
        )
        hypo_logprobs /= hypos_len ** self.args.rescore_length_penalty

        # 3) Compute rewards from rescoring models
        with torch.no_grad():
            rescorer = Rescorer(self.args, self.task, self.rescore_models)
            scores = rescorer.score(src_tokens, hypos)
            rewards = self.combine_score(src_tokens, hypos, hypos_len, scores)
        assert not rewards.requires_grad, "no grads flow back to generation"

        # 4) Compute Policy Gradient loss
        rewards = rewards.type_as(hypo_logprobs)
        rl_loss = -1.0 * (hypo_logprobs * rewards).sum()

        # 5) Compute Cross-entropy loss
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
        nll_loss = -1.0 * target_logprobs.sum()

        # 6) Gather losses
        loss = self.args.rl_weight * rl_loss + nll_loss

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

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        return {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            "nll_loss": loss_sum / ntokens,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

    def combine_score(self, src_tokens, hypos, hypos_len, scores):
        """ Rescore translations and combine weights to find top hypo tokens
        """
        # Prepare all the weights and call combine weighted scores
        args = self.args
        weights = [
            args.l2r_model_weight,
            args.r2l_model_weight,
            args.reverse_model_weight,
            args.lm_model_weight,
            args.cloze_transformer_weight,
        ]
        bsz, src_len = src_tokens.size()
        hypos_len = hypos_len.type_as(scores)
        combined_scores = combine_weighted_scores(
            scores, weights, src_len, hypos_len, args.length_penalty
        )
        return combined_scores

    def load_rescore_models(self, args):
        """load rescoring models"""
        models = {}
        if args.l2r_model_path:
            l2r_model, _, l2r_task = pytorch_translate_utils.load_diverse_ensemble_for_inference(
                [args.l2r_model_path]
            )
            models["l2r_model"] = {"model": l2r_model[0], "task": l2r_task}
        #
        if args.r2l_model_path:
            r2l_model, _, r2l_task = pytorch_translate_utils.load_diverse_ensemble_for_inference(
                [args.r2l_model_path]
            )
            models["r2l_model"] = {"model": r2l_model[0], "task": r2l_task}
        #
        if args.reverse_model_path:
            reverse_model, _, reverse_task = pytorch_translate_utils.load_diverse_ensemble_for_inference(
                [args.reverse_model_path]
            )
            models["reverse_model"] = {"model": reverse_model[0], "task": reverse_task}
        #
        if args.lm_model_path:
            lm_model, _, lm_task = pytorch_translate_utils.load_diverse_ensemble_for_inference(
                [args.lm_model_path]
            )
            models["lm_model"] = {"model": lm_model[0], "task": lm_task}
        #
        if args.cloze_transformer_path:
            cloze_model, _, cloze_task = pytorch_translate_utils.load_diverse_ensemble_for_inference(
                [args.cloze_transformer_path]
            )
            models["cloze_model"] = {"model": cloze_model[0], "task": cloze_task}
        return models
