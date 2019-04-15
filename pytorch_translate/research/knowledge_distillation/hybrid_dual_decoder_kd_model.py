#!/usr/bin/env python3

from fairseq.models import FairseqModel, register_model, register_model_architecture
from pytorch_translate import (
    hybrid_transformer_rnn,
    transformer as pytorch_translate_transformer,
)
from pytorch_translate.utils import torch_find


@register_model("hybrid_dual_decoder_kd")
class HybridDualDecoderKDModel(FairseqModel):
    def __init__(self, task, encoder, teacher_decoder, student_decoder):
        super().__init__(encoder, student_decoder)
        self.teacher_decoder = teacher_decoder
        self.student_decoder = student_decoder
        self.using_teacher = True
        self.task = task

    def get_teacher_model(self):
        return hybrid_transformer_rnn.HybridTransformerRNNModel(
            self.task, self.encoder, self.teacher_decoder
        )

    def get_student_model(self):
        return hybrid_transformer_rnn.HybridTransformerRNNModel(
            self.task, self.encoder, self.student_decoder
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        # command-line args for hybrid_transformer_rnn model are used to build
        # encoder and teacher decoder
        hybrid_transformer_rnn.HybridTransformerRNNModel.add_args(parser)

        # distinct args for student decoder
        parser.add_argument(
            "--student-decoder-embed-dim",
            type=int,
            metavar="N",
            help="[student RNN] decoder embedding dimension",
        )
        parser.add_argument(
            "--student-decoder-layers",
            type=int,
            metavar="N",
            help="[student RNN] num decoder layers",
        )
        parser.add_argument(
            "--student-decoder-attention-heads",
            type=int,
            metavar="N",
            help="[student RNN] num decoder attention heads",
        )
        parser.add_argument(
            "--student-decoder-lstm-units",
            type=int,
            metavar="N",
            help="[student RNN] num LSTM units for each decoder layer",
        )
        parser.add_argument(
            "--student-decoder-out-embed-dim",
            type=int,
            metavar="N",
            help="[student RNN] decoder output embedding dimension",
        )
        parser.add_argument(
            "--student-decoder-reduced-attention-dim",
            type=int,
            default=None,
            metavar="N",
            help="if specified, computes attention with this dimensionality "
            "in the student decoder (instead of using encoder output dims)",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        base_architecture(args)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        encoder_embed_tokens = pytorch_translate_transformer.build_embedding(
            dictionary=src_dict,
            embed_dim=args.encoder_embed_dim,
            path=args.encoder_pretrained_embed,
            freeze=args.encoder_freeze_embed,
        )

        teacher_decoder_embed_tokens = pytorch_translate_transformer.build_embedding(
            dictionary=tgt_dict, embed_dim=args.decoder_embed_dim
        )

        student_decoder_embed_tokens = pytorch_translate_transformer.build_embedding(
            dictionary=tgt_dict, embed_dim=args.student_decoder_embed_dim
        )

        encoder = pytorch_translate_transformer.TransformerEncoder(
            args, src_dict, encoder_embed_tokens, proj_to_decoder=False
        )

        teacher_decoder = hybrid_transformer_rnn.HybridRNNDecoder(
            args, src_dict, tgt_dict, teacher_decoder_embed_tokens
        )

        student_decoder = StudentHybridRNNDecoder(
            args, src_dict, tgt_dict, student_decoder_embed_tokens
        )

        return HybridDualDecoderKDModel(
            task=task,
            encoder=encoder,
            teacher_decoder=teacher_decoder,
            student_decoder=student_decoder,
        )

    def get_targets(self, sample, net_output):
        targets = sample["target"].view(-1)
        possible_translation_tokens = net_output[-1]
        if possible_translation_tokens is not None:
            targets = torch_find(
                possible_translation_tokens, targets, len(self.task.target_dictionary)
            )
        return targets


class StudentHybridRNNDecoder(hybrid_transformer_rnn.HybridRNNDecoder):
    """
    Subclass which constructs RNN decoder from student arguments.
    (dropout, attention_dropout, and vocab reduction params shared with teacher.)
    """

    def _init_dims(self, args, src_dict, dst_dict, embed_tokens):
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.embed_tokens = embed_tokens

        self.lstm_units = args.student_decoder_lstm_units
        self.num_layers = args.student_decoder_layers
        self.initial_input_dim = embed_dim

        self.encoder_output_dim = args.encoder_embed_dim
        if args.student_decoder_reduced_attention_dim is None:
            self.attention_dim = self.encoder_output_dim
        else:
            self.attention_dim = args.student_decoder_reduced_attention_dim
        self.input_dim = self.lstm_units + self.attention_dim

        self.num_attention_heads = args.student_decoder_attention_heads
        self.out_embed_dim = args.student_decoder_out_embed_dim


@register_model_architecture("hybrid_dual_decoder_kd", "hybrid_dual_decoder_kd")
def base_architecture(args):
    hybrid_transformer_rnn.base_architecture(args)
    args.student_decoder_embed_dim = getattr(args, "student_decoder_embed_dim", 128)
    args.student_decoder_layers = getattr(args, "student_decoder_layers", 3)
    args.student_decoder_attention_heads = getattr(
        args, "student_decoder_attention_heads", 8
    )
    args.student_decoder_lstm_units = getattr(args, "student_decoder_lstm_units", 128)
    args.student_decoder_out_embed_dim = getattr(
        args, "student_decoder_out_embed_dim", 128
    )
