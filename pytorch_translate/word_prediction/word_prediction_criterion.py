#!/usr/bin/env python3

import abc
import math

import torch
from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
)
from pytorch_translate.utils import maybe_cuda


class _BasePredictionCriterion(abc.ABC, LabelSmoothedCrossEntropyCriterion):
    """Base class for the losses in order to combine commonly used methods."""

    @abc.abstractmethod
    def predictor_loss_function(self, prediction, target, *args, **kwargs):
        """Pure abstract method that computes the loss.

        Args:
            prediction: Prediction that was made by the model of shape
                        [BATCH_SIZE, N_LABELS]
            target: Expected result of shape [BATCH_SIZE, N_OUTPUT_TOKENS]
        Returns:
            loss: This method should return the loss as a Tensor or Variable.
        """
        return torch.Tensor(float("Inf"))

    def forward(self, model, sample, reduce=True, *args, **kwargs):
        """Computes the loss for the given sample.

        This method uses the inheriting classes' `predictor_loss_function`.

        Args:
            model: Model to use for the loss computation.
            sample: Chosen sample as a dict with at least the following keys:
                    'net_input', 'target', 'ntokens'
            reduce: Boolean flag to reduce the result to per batch elements.
            args, kwargs: Positional/Keyword arguments are passed through to the
                          `predictor_loss_function`.
        Returns:
            loss: Total loss as a Variable
            sample_size: Sample size - used for the gradient denominator.
            logging_output: Logging outputs to display during training.

        Raises:
            AssertionError:
                - prediction and target batch numbers are different
                - prediction shape is not [BATCH_SIZE, N_LABELS]
                - losses for translation and prediction are not the same shape
        """
        predictor_output, decoder_output = model(**sample["net_input"])
        # translation loss
        translation_loss, nll_loss = super().compute_loss(
            model, decoder_output, sample, reduce
        )
        prediction_target = model.get_target_words(sample)
        # predictor loss
        prediction_lprobs = model.get_predictor_normalized_probs(
            predictor_output, log_probs=True
        )
        prediction_lprobs = prediction_lprobs.view(-1, prediction_lprobs.size(-1))

        assert prediction_lprobs.size(0) == prediction_target.size(0)
        assert prediction_lprobs.dim() == 2

        prediction_loss = self.predictor_loss_function(
            prediction_lprobs, prediction_target, *args, **kwargs
        )

        # prevent domination of padding idx
        non_pad_mask = prediction_target.ne(model.encoder.padding_idx)
        prediction_loss = prediction_loss[non_pad_mask]

        # TODO: normalize , sentence avg
        if reduce:
            prediction_loss = prediction_loss.sum()
        else:
            prediction_loss = prediction_loss.sum(1)  # loss per batch element

        assert translation_loss.size() == prediction_loss.size()
        loss = translation_loss + prediction_loss

        if self.sentence_avg:
            sample_size = sample["target"].size(0)
        else:
            sample_size = sample["ntokens"]

        logging_output = {
            "nll_loss": nll_loss,
            "translation_loss": translation_loss.data,
            "prediction_loss": prediction_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if reduce:
            logging_output["translation_loss"] = utils.item(
                logging_output["translation_loss"]
            )
            logging_output["prediction_loss"] = utils.item(
                logging_output["prediction_loss"]
            )
            logging_output["nll_loss"] = utils.item(logging_output["nll_loss"])
        logging_output["loss"] = utils.item(logging_output["translation_loss"])

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregates logging outputs from data parallel training.

        Args:
            logging_outputs: Output log with 'translation_loss' and
                             'prediction_loss'.
        Returns:
            agg_output: Aggregated logs.
        """
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for loss in ["translation_loss", "prediction_loss"]:
            loss_sum = sum(log.get(loss, 0) for log in logging_outputs)

            agg_output[loss] = loss_sum / sample_size / math.log(2)
            if loss == "translation_loss" and sample_size != ntokens:
                agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)

        return agg_output


@register_criterion("word_prediction")
class WordPredictionCriterion(_BasePredictionCriterion):
    """Implements a combined loss from translation and target words prediction."""

    def predictor_loss_function(self, prediction, target):
        """Loss function that maximizes the confidence of the true positive.

        Args:
            prediction: Prediction that was made by the model of shape
                        [BATCH_SIZE, N_LABELS]
            target: Expected result of shape [BATCH_SIZE, N_OUTPUT_TOKENS]

        Returns:
            loss: Loss as a torch.Variable
        """
        return -(prediction.gather(dim=-1, index=target))


@register_criterion("warp_loss")
class WARPLossCriterion(_BasePredictionCriterion):
    """Implements a combined loss from translation and target words prediction.

    References:
      [1] https://research.google.com/pubs/archive/37180.pdf
      [2] https://arxiv.org/abs/1708.01771
      [3] https://discuss.pytorch.org/t/writing-warp-loss-layer/3715

    TODO(T38581791):
      Although this implementation is faster than [3] (3s/it vs 70s/it), it is
      still much slower than `WordPredictionCriterion`.
      Need to explore ways to speed it up -- it looks like the training time
      increases after each iteration. This happens because it becomes harder
      to search after each update.
    """

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task, sentence_avg, label_smoothing)
        self.rank_weights = 0.0

    def set_rank_weights(self, n_labels, rank_weights_type="uniform"):
        """Sets ranking for weights based on the number of labels.

        Args:
            n_labels: Number of labels
            rank_weights_type: Type of the ranking.

        Raises:
            AssertionError: Number of labels <= 1
            NotImplementedError: rank_weights_type is not 'uniform'
        """
        assert n_labels > 1
        if rank_weights_type == "uniform":
            self.rank_weights = 1.0 / (n_labels - 1) * maybe_cuda(torch.ones(n_labels))
        else:
            raise NotImplementedError(
                "Rank weights type {} not implemented".format(rank_weights_type)
            )

    def predictor_loss_function(self, prediction, target, rank_weights_type="uniform"):
        """Implements the WARP loss given in [1].

        In its core the function computes the following:
            loss = (X-1)/N*(xn_i - xp),
        where `xn_i` is confidence of the ith false positive, and `xp` is the
        true positive confidence. `X` is the total number of labels and `N` is
        the number of steps that it takes to find a false positive.
        Note: We might want to use ln((X-1)/N), in case N << X, which would
              expolode the loss.

        Args:
            prediction: Prediction that was made by the model of shape
                        [BATCH_SIZE, N_LABELS]
            target: Expected result of shape [BATCH_SIZE, N_OUTPUT_TOKENS]
            rank_weight_type: Argument to set the ranks of the weights.
                              See `set_rank_weights` for more details.

        Returns:
            loss: Loss as a torch.Variable
        """
        batch_size = prediction.size()[0]
        n_labels = prediction.size()[1]
        n_output_tokens = target.size()[1]
        max_num_trials = n_labels - 1

        self.set_rank_weights(n_labels, rank_weights_type)

        loss = maybe_cuda(torch.zeros(batch_size, n_output_tokens))

        for i in range(batch_size):
            for j in range(n_output_tokens):
                target_idx = target[i, j]
                neg_labels_idx = maybe_cuda(
                    torch.tensor(
                        list(set(range(n_labels)) - set(target[i, :].cpu().numpy()))
                    )
                )
                neg_idx = torch.multinomial(neg_labels_idx.double(), 1)
                # This is the hinge loss:
                # sample_score_margin = \
                #   1 - prediction[i, target_idx] + prediction[i, neg_idx]
                # TODO:
                #   Since |- prediction[i, target_idx] + prediction[i, neg_idx]|
                #   is normally around 0.01, directly using log probability in
                #   hinge loss causes most N to be 1, thus is not a good choice.
                # Observation: translation_loss is normally ~10, similar to
                #              log_probs.
                # Alternatives: scale up score difference by 100 times to match
                #               the magnitude of 1, but we also need to consider
                #               magnitude of weights and loss;
                sample_score_margin = (
                    -prediction[i, target_idx] + prediction[i, neg_idx]
                )
                N = 1
                while sample_score_margin < 0 and N < max_num_trials:
                    neg_idx = torch.multinomial(neg_labels_idx.double(), 1)
                    N += 1
                    sample_score_margin = (
                        -prediction[i, target_idx] + prediction[i, neg_idx]
                    )

                k = torch.floor(torch.tensor(max_num_trials / N)).int()
                weights = torch.sum(self.rank_weights[:k])
                score_margins = -prediction[i, target_idx] + prediction[i, neg_idx]
                loss[i, j] = (weights * score_margins.clamp(min=0.0)).mean()
        return loss
