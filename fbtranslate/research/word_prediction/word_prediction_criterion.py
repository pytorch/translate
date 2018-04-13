import math
import torch
import torch.nn.functional as F

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import utils


@register_criterion('word_prediction')
class WordPredictionCriterion(FairseqCriterion):

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        predictor_output, decoder_output = model(**sample['net_input'])
        # translation loss
        translation_lprobs = model.get_normalized_probs(decoder_output, log_probs=True)
        translation_target = model.get_targets(sample, decoder_output).view(-1)
        translation_loss = F.nll_loss(
            translation_lprobs,
            translation_target,
            size_average=False,
            ignore_index=self.padding_idx,
            reduce=reduce
        )
        # predictor loss
        prediction_lprobs = model.get_predictor_normalized_probs(
            predictor_output, log_probs=True)
        # prevent domination of padding idx
        non_padding_mask = torch.ones(prediction_lprobs.size(1)).cuda()
        non_padding_mask[model.encoder.padding_idx] = 0
        prediction_lprobs = prediction_lprobs * non_padding_mask.unsqueeze(0)

        prediction_target = model.get_target_words(sample)
        assert prediction_lprobs.size(0) == prediction_target.size(0)
        assert prediction_lprobs.dim() == 2
        word_prediction_loss = -torch.gather(prediction_lprobs, 1, prediction_target)

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
            'loss': translation_loss.data,
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

        for loss in ['loss', 'word_prediction_loss']:
            loss_sum = sum(log.get(loss, 0) for log in logging_outputs)

            agg_output[loss] = loss_sum / sample_size / math.log(2)
            if loss == 'loss' and sample_size != ntokens:
                agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)

        return agg_output
