#!/usr/bin/env python3

import copy
import math

import torch
from fairseq import utils
from fairseq.criterions import LegacyFairseqCriterion, register_criterion
from fairseq.scoring import bleu
from pytorch_translate import generate


"""Sequence-level losses from Edunov et al., 2017
(https://arxiv.org/pdf/1711.04956.pdf).
"""


class BaseSequenceLossCriterion(LegacyFairseqCriterion):
    """Base class for criteria with need to run beam search."""

    def __init__(self, args, src_dict, dst_dict):
        super().__init__(args, src_dict, dst_dict)
        self.translator = None
        self.scorer = bleu.Scorer(
            bleu.BleuConfig(
                pad=dst_dict.pad(),
                eos=dst_dict.eos(),
                unk=dst_dict.unk(),
            )
        )

    def get_translator(self, model):
        """Get lazy singleton translator instance."""
        if self.translator is None:
            args_clone = copy.copy(self.args)
            if self.args.loss_beam:  # Override beam size if necessary
                args_clone.beam = self.args.loss_beam
            self.translator = generate.build_sequence_generator(args_clone, [model])
        return self.translator

    def generate_translations(self, model, sample):
        """Run beam search to generate translations from the current model.

        Args:
            model: FairseqModel to use (passed via FairseqCriterion.forward())
            sample: Training batch (passed via FairseqCriterion.forward())

        Returns:
            A tuple (tokens, bleu) of tensors. `tokens` is a [bsz, beam_size,
            max_translation_length] int tensor with generated translations (with
            EOS). `bleu` is a [bsz, beam_size] float tensor with BLEU scores.
            `tokens` corresponds to U(x) in Edunov et al., 2017.
        """
        model.eval()  # Set to eval mode
        enc_input = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        trg_lens = sample["target"].size(1) - torch.sum(
            sample["target"] <= self.scorer.eos, dim=1
        )
        max_trans_len = 0
        all_translations = []
        for batch_idx, batch_trans in enumerate(
            self._batch_translations(model, enc_input)
        ):
            ref = sample["target"][batch_idx, : trg_lens[batch_idx] + 1].int().cpu()
            batch_translations = []
            for trans in batch_trans:
                max_trans_len = max(max_trans_len, trans["tokens"].size(0))
                self.scorer.reset()
                self.scorer.add(ref, trans["tokens"].int().cpu())
                batch_translations.append((self.scorer.score(), trans["tokens"]))
            batch_translations.sort(key=lambda x: -x[0])
            all_translations.append(batch_translations)
        bsz = len(all_translations)
        beam_size = self.get_translator(model).beam_size
        tokens_tensor = trans["tokens"].new_full(
            (bsz, beam_size, max_trans_len), self.scorer.pad
        )
        bleu_tensor = torch.zeros(bsz, beam_size)
        for batch_idx, batch_translations in enumerate(all_translations):
            for trans_idx, (bleu_score, tokens) in enumerate(batch_translations):
                tokens_tensor[batch_idx, trans_idx, : tokens.size(0)] = tokens
                bleu_tensor[batch_idx, trans_idx] = bleu_score
        model.train()  # Set back to train mode
        return tokens_tensor, bleu_tensor

    def _batch_translations(self, model, enc_input):
        """Use beam_size times smaller batch size for translation."""
        translator = self.get_translator(model)
        bsz = enc_input[1].size(0)
        gen_bsz = bsz // translator.beam_size + 1
        for f in range(0, bsz, gen_bsz):
            t = min(f + gen_bsz, bsz)
            for trans in translator.generate((enc_input[0][f:t], enc_input[1][f:t])):
                yield trans

    def compute_nll(self, model, sample, translations):
        """Compute negative log-likelihoods for the translations.

        This function computes p(u|x) for each element in U(x) (see Edunov et
        al., 2017 for notation) for computing gradients.

        Args:
            model: FairseqModel to use (passed via FairseqCriterion.forward())
            sample: Training batch (passed via FairseqCriterion.forward())
            translations: A [batch_size, beam_size, max_trg_len] int tensor of
                generated translation (as produced by generate_translations())

        Returns:
            A [bsz, beam_size] float tensor containing the negative loglikelihoods
            of the sentences in `translations`.
        """
        bsz, beam_size, max_trans_len = translations.size()
        total = bsz * beam_size
        translations = translations.view(total, max_trans_len)
        prev_output_tokens = translations.new(total, max_trans_len)
        prev_output_tokens[:, 0] = self.scorer.eos
        prev_output_tokens[:, 1:] = translations[:, :-1]
        src_tokens = (
            sample["net_input"]["src_tokens"]
            .unsqueeze(1)
            .repeat(1, beam_size, 1)
            .view(total, -1)
        )
        src_lengths = (
            sample["net_input"]["src_lengths"]
            .unsqueeze(1)
            .repeat(1, beam_size)
            .view(total)
        )
        all_losses = []
        for f in range(0, total, bsz):
            t = min(f + bsz, total)
            net_output = model(
                src_tokens=src_tokens[f:t],
                src_lengths=src_lengths[f:t],
                prev_output_tokens=prev_output_tokens[f:t],
            )
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            nll_loss = -(lprobs.gather(dim=-1, index=translations[f:t].unsqueeze(2)))
            non_pad_mask = translations[f:t].ne(self.padding_idx).float()
            all_losses.append(torch.sum(nll_loss.squeeze(2) * non_pad_mask, dim=1))
        return torch.cat(all_losses).view(bsz, beam_size)

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        return {
            "loss": sum(log.get("loss", 0) for log in logging_outputs)
            / sample_size
            / math.log(2),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }


@register_criterion("sequence_nll")
class SequenceNegativeLoglikelihoodCriterion(BaseSequenceLossCriterion):
    """SeqNLL loss from https://arxiv.org/pdf/1711.04956.pdf."""

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        translations, bleu_scores = self.generate_translations(model, sample)
        nll_loss = self.compute_nll(model, sample, translations)
        loss = nll_loss[:, 0] + torch.logsumexp(-nll_loss, 1)
        if reduce:
            loss = loss.sum()
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


@register_criterion("sequence_risk")
class SequenceRiskCriterion(BaseSequenceLossCriterion):
    """Risk loss from https://arxiv.org/pdf/1711.04956.pdf."""

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        translations, bleu_scores = self.generate_translations(model, sample)
        nll_loss = self.compute_nll(model, sample, translations)
        partition = torch.logsumexp(-nll_loss, 1)
        probs = torch.exp(-nll_loss - partition.unsqueeze(1))
        loss = torch.sum((1 - bleu_scores.cuda() / 100) * probs, dim=1)
        if reduce:
            loss = loss.sum()
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
