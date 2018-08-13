#!/usr/bin/env python3

from fairseq.models import FairseqModel, register_model, register_model_architecture
from pytorch_translate import rnn
from pytorch_translate.rnn import (
    LSTMSequenceEncoder,
    RNNDecoder,
    RNNEncoder,
    torch_find,
)
from pytorch_translate.word_prediction import word_predictor


class WordPredictionModel(FairseqModel):
    """
    An architecuture which jointly learns translation and target words
    prediction, as described in http://aclweb.org/anthology/D17-1013.
    """

    def __init__(self, task, encoder, decoder, predictor):
        super().__init__(encoder, decoder)
        self.predictor = predictor
        self.task = task

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_output = self.encoder(src_tokens, src_lengths)
        pred_output = self.predictor(encoder_output)
        decoder_output = self.decoder(prev_output_tokens, encoder_output)
        return pred_output, decoder_output

    def get_predictor_normalized_probs(self, pred_output, log_probs):
        return self.predictor.get_normalized_probs(pred_output, log_probs)

    def get_target_words(self, sample):
        return sample["target"]


@register_model("rnn_word_pred")
class RNNWordPredictionModel(WordPredictionModel):
    """
    A subclass which adds words prediction to RNN arch.
    """

    @staticmethod
    def add_args(parser):
        rnn.RNNModel.add_args(parser)
        parser.add_argument(
            "--predictor-hidden-dim",
            type=int,
            metavar="N",
            help="word predictor num units",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        src_dict, dst_dict = task.source_dictionary, task.target_dictionary
        base_architecture_wp(args)
        if args.sequence_lstm:
            encoder_class = LSTMSequenceEncoder
        else:
            encoder_class = RNNEncoder
        decoder_class = RNNDecoder

        encoder = encoder_class(
            src_dict,
            embed_dim=args.encoder_embed_dim,
            freeze_embed=args.encoder_freeze_embed,
            cell_type=args.cell_type,
            num_layers=args.encoder_layers,
            hidden_dim=args.encoder_hidden_dim,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            residual_level=args.residual_level,
            bidirectional=bool(args.encoder_bidirectional),
        )
        decoder = decoder_class(
            src_dict=src_dict,
            dst_dict=dst_dict,
            vocab_reduction_params=args.vocab_reduction_params,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            freeze_embed=args.decoder_freeze_embed,
            out_embed_dim=args.decoder_out_embed_dim,
            cell_type=args.cell_type,
            num_layers=args.decoder_layers,
            hidden_dim=args.decoder_hidden_dim,
            attention_type=args.attention_type,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            residual_level=args.residual_level,
            averaging_encoder=args.averaging_encoder,
        )
        predictor = word_predictor.WordPredictor(
            args.encoder_hidden_dim, args.predictor_hidden_dim, len(dst_dict)
        )
        return cls(task, encoder, decoder, predictor)

    def get_targets(self, sample, net_output):
        targets = sample["target"].view(-1)
        possible_translation_tokens = net_output[-1]
        if possible_translation_tokens is not None:
            targets = torch_find(
                possible_translation_tokens, targets, len(self.task.target_dictionary)
            )
        return targets


@register_model_architecture("rnn_word_pred", "rnn_word_pred")
def base_architecture_wp(args):
    # default architecture
    rnn.base_architecture(args)
    args.predictor_hidden_dim = getattr(args, "predictor_hidden_dim", 512)
