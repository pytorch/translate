#!/usr/bin/env python3

import torch
import torch.nn as nn
from fairseq import tasks
from fairseq.models import (
    ARCH_MODEL_REGISTRY,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
    transformer as fairseq_transformer,
)
from pytorch_translate import (
    transformer as pytorch_translate_transformer,
    vocab_reduction,
)


@register_model("vocab_prediction")
class VocabPredictionModel(FairseqEncoderDecoderModel):
    """
    This is a notional "sequence-to-sequence" which is actually used to
    train a specialized module used to predict a subset of vocabulary tokens
    for another previously trained seq2seq model. The encoder of that other
    model is frozen and its outputs are used to predict a reduced pool of
    possible output tokens for each input sequence.
    """

    def __init__(self, task, encoder, decoder, detach_encoder_updates: bool = False):
        super().__init__(encoder, decoder)
        self.task = task
        self.detach_encoder_updates = detach_encoder_updates

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pytorch_translate_transformer.TransformerModel.add_args(parser)

        parser.add_argument(
            "--seq2seq-model-checkpoint",
            type=str,
            help="checkpoint for existing seq2seq model for which to learn "
            "vocab prediction model",
        )
        parser.add_argument(
            "--detach-encoder-updates",
            default=False,
            action="store_true",
            help="If true, detach model updates.",
        )
        parser.add_argument(
            "--num-top-tokens",
            type=int,
            default=2000,
            metavar="D",
            help="number of fixed top-frequency tokens to always include.",
        )
        parser.add_argument(
            "--topk-labels-per-source-token",
            type=int,
            default=30,
            metavar="D",
            help="number of target tokens to predict for each source token.",
        )
        parser.add_argument(
            "--vocab-reduction-weight",
            type=float,
            default=0.5,
            metavar="D",
            help="In case of joint learning,  weight for vocab reduction loss.",
        )
        parser.add_argument(
            "--core-arch",
            type=str,
            default="ptt_transformer",
            metavar="ARCH",
            help="Core architecture of the model.",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        base_architecture(args)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if hasattr(args, "seq2seq_model_checkpoint"):
            # load existing seq2seq model (with which vocab porediction shares
            # encoder)
            checkpoint_data = torch.load(args.seq2seq_model_checkpoint)
            checkpoint_args = checkpoint_data["args"]

            checkpoint_task = tasks.setup_task(checkpoint_args)
            core_model = checkpoint_task.build_model(checkpoint_args)
            core_model.load_state_dict(checkpoint_data["model"])
            print("loaded the pratrained model!")
        else:
            core_model = ARCH_MODEL_REGISTRY[args.core_arch].build_model(args, task)

        print("type of core_model", type(core_model))

        detach_encoder_updates = hasattr(args, "detach_encoder_updates")

        encoder = core_model.encoder
        decoder = core_model.decoder

        vocab_predictor = VocabPredictor(args, tgt_dict)
        decoder.vocab_reduction_module = vocab_reduction.VocabReduction(
            src_dict,
            tgt_dict,
            vocab_reduction_params=None,
            predictor=vocab_predictor,
            fp16=args.fp16,
            use_during_training=False,
        )

        return VocabPredictionModel(
            task=task,
            encoder=encoder,
            decoder=decoder,
            detach_encoder_updates=detach_encoder_updates,
        )


class VocabPredictor(nn.Module):
    """
    Predicts subset of overall vocabulary based on encoder output.
    """

    def __init__(self, args, tgt_dict):
        super().__init__()

        self.num_top_tokens = args.num_top_tokens
        self.topk_labels_per_source_token = args.topk_labels_per_source_token

        self.dropout = args.dropout
        self.hidden_layers = nn.ModuleList([])
        self.hidden_layers.append(
            fairseq_transformer.Linear(args.encoder_embed_dim, args.hidden_dim)
        )
        for _ in range(args.num_hidden_layers - 1):
            self.hidden_layers.append(
                fairseq_transformer.Linear(args.hidden_dim, args.hidden_dim)
            )

        assert self.num_top_tokens < len(tgt_dict)
        self.output_dim = len(tgt_dict) - self.num_top_tokens
        self.output_projection = fairseq_transformer.Linear(
            args.hidden_dim, self.output_dim
        )

    def forward(self, encoder_outputs, encoder_padding_mask=None):
        x = encoder_outputs
        if type(encoder_outputs) is tuple:
            x = encoder_outputs[0]

        residual = None
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)
            x = torch.dropout(x, self.dropout, self.training)
            if residual is not None:
                x = x + residual
            residual = x

        sequence_output = self.output_projection(x)
        if encoder_padding_mask is not None:
            sequence_output[encoder_padding_mask.t()] = -float("inf")

        # max pooling over axis 0 (sequence dimension)
        pooled_output, _ = sequence_output.max(0)

        return pooled_output

    def get_topk_predicted_tokens(self, net_output, src_tokens, log_probs: bool):
        """
        Get self.topk_labels_per_source_token top predicted words for vocab
        reduction (per source token).
        """
        k = min(src_tokens.size(1) * self.topk_labels_per_source_token, self.output_dim)

        # always include most frequent num_top_tokens
        _, topk_offsets = torch.topk(net_output, k, dim=1)
        topk_indices = topk_offsets + self.num_top_tokens
        fixed_tokens = torch.arange(self.num_top_tokens)
        fixed_tokens = fixed_tokens.unsqueeze(0).repeat(net_output.size(0), 1)
        fixed_tokens = fixed_tokens.type_as(topk_indices)

        prediction = torch.cat([fixed_tokens, topk_indices], dim=1)
        return prediction


@register_model_architecture("vocab_prediction", "vocab_prediction")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.2)
    args.hidden_dim = getattr(args, "hidden_dim", 4096)
    args.num_hidden_layers = getattr(args, "num_hidden_layers", 2)
    args.topk_labels_per_source_token = getattr(args, "topk_labels_per_source_token", 2)
