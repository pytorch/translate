#!/usr/bin/env python3

import math

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy \
    import LabelSmoothedCrossEntropyCriterion
from fairseq import utils


@register_criterion('word_prediction')
class WordPredictionCriterion(LabelSmoothedCrossEntropyCriterion):
    """
    Implement a combined loss from translation and target words prediction.
    """
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) total loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        predictor_output, decoder_output = model(**sample['net_input'])
        # translation loss
        translation_loss, _ = super().compute_loss(
            model,
            decoder_output,
            sample,
            reduce,
        )
        prediction_target = model.get_target_words(sample)
        # predictor loss
        prediction_lprobs = model.get_predictor_normalized_probs(
            predictor_output, log_probs=True)
        prediction_lprobs = prediction_lprobs.view(-1, prediction_lprobs.size(-1))
        # prevent domination of padding idx
        non_pad_mask = prediction_target.ne(model.encoder.padding_idx)

        assert prediction_lprobs.size(0) == prediction_target.size(0)
        assert prediction_lprobs.dim() == 2
        word_prediction_loss = -prediction_lprobs.gather(
            dim=-1,
            index=prediction_target,
        )[non_pad_mask]
        # TODO: normalize , sentence avg
        if reduce:
            word_prediction_loss = word_prediction_loss.sum()
        else:
            word_prediction_loss = word_prediction_loss.sum(1)  # loss per batch element

        assert translation_loss.size() == word_prediction_loss.size()
        loss = translation_loss + word_prediction_loss

        if self.args.sentence_avg:
            sample_size = sample['target'].size(0)
        else:
            sample_size = sample['ntokens']

        logging_output = {
            'translation_loss': translation_loss.data,
            'word_prediction_loss': word_prediction_loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }

        if reduce:
            logging_output['loss'] = utils.item(logging_output['loss'])
            logging_output['word_prediction_loss'] = utils.item(
                logging_output['word_prediction_loss'])

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {'sample_size': sample_size}

        for loss in ['translation_loss', 'word_prediction_loss']:
            loss_sum = sum(log.get(loss, 0) for log in logging_outputs)

            agg_output[loss] = loss_sum / sample_size / math.log(2)
            if loss == 'translation_loss' and sample_size != ntokens:
                agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)

        return agg_output
