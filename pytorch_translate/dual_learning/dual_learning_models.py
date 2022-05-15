#!/usr/bin/env python3

import logging

import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.models import BaseFairseqModel, register_model
from pytorch_translate import rnn
from pytorch_translate.rnn import (
    base_architecture,
    LSTMSequenceEncoder,
    RNNDecoder,
    RNNEncoder,
    RNNModel,
)
from pytorch_translate.tasks.pytorch_translate_task import PytorchTranslateTask


logger = logging.getLogger(__name__)


@register_model("dual_learning")
class DualLearningModel(BaseFairseqModel):
    """
    An architecture to jointly train primal model and dual model by leveraging
    distribution duality, which exist for both parallel data and monolingual
    data.
    """

    def __init__(self, args, task, primal_model, dual_model, lm_model=None):
        super().__init__()
        self.args = args
        self.task_keys = ["primal", "dual"]
        self.models = nn.ModuleDict(
            {"primal": primal_model, "dual": dual_model, "lm": lm_model}
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens=None):
        """
        If batch is monolingual, need to run beam decoding to generate
        fake prev_output_tokens.
        """

        # TODO: pass to dual model too
        primal_encoder_out = self.models["primal"].encoder(src_tokens, src_lengths)
        primal_decoder_out = self.models["primal"].decoder(
            prev_output_tokens, primal_encoder_out
        )
        return primal_decoder_out

    def max_positions(self):
        return {
            "primal_source": (
                self.models["primal"].encoder.max_positions(),
                self.models["primal"].decoder.max_positions(),
            ),
            "dual_source": (
                self.models["dual"].encoder.max_positions(),
                self.models["dual"].decoder.max_positions(),
            ),
            "primal_parallel": (
                self.models["primal"].encoder.max_positions(),
                self.models["primal"].decoder.max_positions(),
            ),
            "dual_parallel": (
                self.models["dual"].encoder.max_positions(),
                self.models["dual"].decoder.max_positions(),
            ),
        }


@register_model("dual_learning_rnn")
class RNNDualLearningModel(DualLearningModel):
    """Train two models for a task and its duality jointly.
    This class uses RNN arch, but can be extended to take arch as an arument.
    This class takes translation as a task, but the framework is intended
    to be general enough to be applied to other tasks as well.
    """

    def __init__(self, args, task, primal_model, dual_model, lm_model=None):
        super().__init__(args, task, primal_model, dual_model, lm_model)

    @staticmethod
    def add_args(parser):
        rnn.RNNModel.add_args(parser)
        parser.add_argument(
            "--unsupervised-dual",
            default=False,
            action="store_true",
            help="Train with dual loss from monolingual data.",
        )
        parser.add_argument(
            "--supervised-dual",
            default=False,
            action="store_true",
            help="Train with dual loss from parallel data.",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build both the primal and dual models.
        For simplicity, both models share the same arch, i.e. the same model
        params would be used to initialize both models.
        Support for different models/archs would be added in further iterations.
        """
        base_architecture(args)

        if args.sequence_lstm:
            encoder_class = LSTMSequenceEncoder
        else:
            encoder_class = RNNEncoder
        decoder_class = RNNDecoder

        encoder_embed_tokens, decoder_embed_tokens = RNNModel.build_embed_tokens(
            args, task.primal_src_dict, task.primal_tgt_dict
        )
        primal_encoder = encoder_class(
            task.primal_src_dict,
            embed_dim=args.encoder_embed_dim,
            embed_tokens=encoder_embed_tokens,
            cell_type=args.cell_type,
            num_layers=args.encoder_layers,
            hidden_dim=args.encoder_hidden_dim,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            residual_level=args.residual_level,
            bidirectional=bool(args.encoder_bidirectional),
        )
        primal_decoder = decoder_class(
            src_dict=task.primal_src_dict,
            dst_dict=task.primal_tgt_dict,
            embed_tokens=decoder_embed_tokens,
            vocab_reduction_params=args.vocab_reduction_params,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
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
        primal_task = PytorchTranslateTask(
            args, task.primal_src_dict, task.primal_tgt_dict
        )
        primal_model = rnn.RNNModel(primal_task, primal_encoder, primal_decoder)
        if args.pretrained_forward_checkpoint:
            pretrained_forward_state = checkpoint_utils.load_checkpoint_to_cpu(
                args.pretrained_forward_checkpoint
            )
            primal_model.load_state_dict(pretrained_forward_state["model"], strict=True)
            print(
                f"Loaded pretrained primal model from {args.pretrained_forward_checkpoint}"
            )

        encoder_embed_tokens, decoder_embed_tokens = RNNModel.build_embed_tokens(
            args, task.dual_src_dict, task.dual_tgt_dict
        )
        dual_encoder = encoder_class(
            task.dual_src_dict,
            embed_dim=args.encoder_embed_dim,
            embed_tokens=encoder_embed_tokens,
            cell_type=args.cell_type,
            num_layers=args.encoder_layers,
            hidden_dim=args.encoder_hidden_dim,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            residual_level=args.residual_level,
            bidirectional=bool(args.encoder_bidirectional),
        )
        dual_decoder = decoder_class(
            src_dict=task.dual_src_dict,
            dst_dict=task.dual_tgt_dict,
            embed_tokens=decoder_embed_tokens,
            vocab_reduction_params=args.vocab_reduction_params,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
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
        dual_task = PytorchTranslateTask(args, task.dual_src_dict, task.dual_tgt_dict)
        dual_model = rnn.RNNModel(dual_task, dual_encoder, dual_decoder)
        if args.pretrained_backward_checkpoint:
            pretrained_backward_state = checkpoint_utils.load_checkpoint_to_cpu(
                args.pretrained_backward_checkpoint
            )
            dual_model.load_state_dict(pretrained_backward_state["model"], strict=True)
            print(
                f"Loaded pretrained dual model from {args.pretrained_backward_checkpoint}"
            )

        # TODO (T36875783): instantiate a langauge model
        lm_model = None
        return RNNDualLearningModel(args, task, primal_model, dual_model, lm_model)
