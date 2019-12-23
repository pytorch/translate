#!/usr/bin/env python3

import torch
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from pytorch_translate.utils import maybe_cuda


@register_criterion("vocab_prediction_loss")
class VocabPredictionCriterion(FairseqCriterion):
    """
    This is a loss for multi-label classification
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.vocab_reduction_weight = getattr(args, "vocab_reduction_weight", 0.5)
        self.eps = 0

        assert (
            not self.args.sentence_avg
        ), "sentence_avg not compatible with VocabPredictionCriterion!"

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
        kwargs = {}
        for k, v in sample["net_input"].items():
            if k not in {"src_lengths", "prev_output_tokens", "src_tokens"}:
                kwargs[k] = v

        encoder_out = model.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # cache vocab_reduction_module to run decoder on full vocab
        vocab_reduction_module = model.decoder.vocab_reduction_module
        model.decoder.vocab_reduction_module = None

        decoder_out = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        translation_loss, nll_loss = self.compute_decoder_loss(
            model, decoder_out, sample, reduce=reduce
        )

        # reattach vocab_reduction_module
        model.decoder.vocab_reduction_module = vocab_reduction_module

        encoder_states_for_vocab_reduction = encoder_out[0]
        if model.detach_encoder_updates:
            encoder_states_for_vocab_reduction = (
                encoder_states_for_vocab_reduction.detach()
            )

        encoder_padding_mask = src_tokens.eq(self.task.src_dict.pad())
        vocab_prediction_output = model.decoder.vocab_reduction_module.predictor(
            encoder_states_for_vocab_reduction,
            encoder_padding_mask=encoder_padding_mask,
        )

        sequence_targets = sample["target"]
        # we always use num_top_tokens: not part of the prediction
        num_top_tokens = model.decoder.vocab_reduction_module.predictor.num_top_tokens
        modified_sequence_targets = sequence_targets - num_top_tokens
        valid_mask = torch.ge(modified_sequence_targets, 0)

        batch_indices = (
            torch.LongTensor(range(modified_sequence_targets.size(0)))
            .unsqueeze(1)
            .repeat(1, modified_sequence_targets.size(1))
        )

        # only use valid tokens (not within num_top_tokens most frequent)
        valid_sequence_targets = modified_sequence_targets[valid_mask]
        batch_indices = batch_indices[valid_mask]

        positive_labels = torch.zeros(vocab_prediction_output.shape)
        positive_labels[batch_indices, valid_sequence_targets] = 1
        positive_labels = maybe_cuda(positive_labels)

        # bounded for numerical stability
        probs = torch.sigmoid(vocab_prediction_output).clamp(min=1e-5, max=0.99999)
        positive_label_loss = -(torch.log(probs) * positive_labels)

        negative_labels = 1 - positive_labels
        negative_label_loss = -(torch.log1p(-probs) * negative_labels)

        # scale up to account for difference in loss scale with translation_loss
        tokens_per_sequence = sample["target"].ne(self.task.tgt_dict.pad()).sum(1)

        # to prevent instability in case of zero (e.g., positive labels within
        # num_top_tokens)
        eps = 1e-5

        # combine separate means for positive and negative loss for each
        # sequence (to skew update toward positive examples)
        num_positive_labels = positive_labels.sum(1)
        scaled_positive_label_loss = positive_label_loss.sum(1) * tokens_per_sequence
        mean_positive_label_loss = scaled_positive_label_loss / (
            num_positive_labels + eps
        )

        num_negative_labels = negative_labels.sum(1)
        scaled_negative_label_loss = negative_label_loss.sum(1) * tokens_per_sequence
        mean_negative_label_loss = scaled_negative_label_loss / (
            num_negative_labels + eps
        )
        total_vocab_prediction_loss = (
            mean_positive_label_loss.sum() + mean_negative_label_loss.sum()
        )
        total_loss = (
            self.vocab_reduction_weight * total_vocab_prediction_loss
            + (1 - self.vocab_reduction_weight) * translation_loss
        )
        # number of tokens
        sample_size = sample["ntokens"]

        logging_output = {
            "vocab_loss": utils.item(total_vocab_prediction_loss.data)
            if reduce
            else total_vocab_prediction_loss.data,
            "nmt_loss": utils.item(translation_loss) if reduce else translation_loss,
            "loss": total_loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return total_loss, sample_size, logging_output

    def compute_decoder_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce
        )
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        return {
            "vocab_loss": sum(log.get("vocab_loss", 0) for log in logging_outputs)
            / sample_size,
            "nmt_loss": sum(log.get("nmt_loss", 0) for log in logging_outputs)
            / sample_size,
            "loss": sum(log.get("loss", 0) for log in logging_outputs) / sample_size,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
