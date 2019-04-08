#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.operators
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax
from pytorch_translate import rnn_cell  # noqa
from pytorch_translate import (
    attention,
    dictionary as pytorch_translate_dictionary,
    utils as pytorch_translate_utils,
    vocab_reduction,
)
from pytorch_translate.common_layers import (
    ContextEmbedding,
    DecoderWithOutputProjection,
    Embedding,
    Linear,
    RNNLayer,
    VariableTracker,
)
from pytorch_translate.data import data as pytorch_translate_data
from pytorch_translate.multi_model import MultiDecoder, MultiEncoder
from pytorch_translate.multilingual import MultilingualDecoder, MultilingualEncoder
from pytorch_translate.ngram import NGramDecoder
from pytorch_translate.semi_supervised import SemiSupervisedModel
from pytorch_translate.utils import maybe_cat, maybe_cuda, torch_find
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


def reorder_encoder_output(encoder_out, new_order):
    """Reorder all outputs according to new_order."""
    (
        unpacked_output,
        final_hiddens,
        final_cells,
        src_lengths,
        src_tokens,
        src_embeddings,
    ) = encoder_out
    unpacked_output = unpacked_output.index_select(1, new_order)
    final_hiddens = final_hiddens.index_select(1, new_order)
    final_cells = final_cells.index_select(1, new_order)
    src_lengths = src_lengths.index_select(0, new_order)
    src_tokens = pytorch_translate_utils.get_source_tokens_tensor(
        src_tokens
    ).index_select(0, new_order)
    src_embeddings = src_embeddings.index_select(1, new_order)
    return (
        unpacked_output,
        final_hiddens,
        final_cells,
        src_lengths,
        src_tokens,
        src_embeddings,
    )


@register_model("rnn")
class RNNModel(FairseqModel):
    def __init__(self, task, encoder, decoder):
        super().__init__(encoder, decoder)
        self.task = task

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--language-model-only",
            default=False,
            action="store_true",
            help="whether to train a language model only where no encoder is used",
        )
        parser.add_argument(
            "--dropout",
            default=0.1,
            type=float,
            metavar="D",
            help="dropout probability",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            default=0,
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-pretrained-embed",
            default=None,
            metavar="FILE",
            help="path to pre-trained encoder embedding",
        )
        parser.add_argument(
            "--encoder-freeze-embed",
            default=False,
            action="store_true",
            help=(
                "whether to freeze the encoder embedding or allow it to be "
                "updated during training"
            ),
        )
        parser.add_argument(
            "--encoder-normalize-embed",
            default=False,
            action="store_true",
            help=(
                "whether to normalize the encoder embeddings to have zero mean "
                "and unit variance (weighted by token frequency)"
            ),
        )
        parser.add_argument(
            "--encoder-hidden-dim", type=int, metavar="N", help="encoder cell num units"
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="number of encoder layers"
        )
        parser.add_argument(
            "--encoder-bidirectional",
            action="store_true",
            help="whether the first layer is bidirectional or not",
        )
        parser.add_argument(
            "--averaging-encoder",
            default=False,
            action="store_true",
            help=(
                "whether use mean encoder hidden states as decoder initial "
                "states or not"
            ),
        )
        parser.add_argument(
            "--decoder-embed-dim",
            default=0,
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-pretrained-embed",
            default=None,
            metavar="FILE",
            help="path to pre-trained decoder embedding",
        )
        parser.add_argument(
            "--decoder-freeze-embed",
            default=False,
            action="store_true",
            help=(
                "whether to freeze the decoder embedding or allow it to be "
                "updated during training"
            ),
        )
        parser.add_argument(
            "--decoder-hidden-dim", type=int, metavar="N", help="decoder cell num units"
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="number of decoder layers"
        )
        parser.add_argument(
            "--decoder-out-embed-dim",
            type=int,
            metavar="N",
            help="decoder output embedding dimension",
        )
        parser.add_argument(
            "--decoder-out-pretrained-embed",
            default=None,
            metavar="FILE",
            help="path to pre-trained decoder output embedding",
        )
        parser.add_argument(
            "--out-embed-norm",
            default=None,
            type=float,
            help="norm for output projection weights",
        )
        parser.add_argument(
            "--decoder-tie-embeddings",
            default=False,
            action="store_true",
            help="tie the decoder word embeddings with the output projection "
            "weights (requires that the embedding dims be of the same size)",
        )
        parser.add_argument(
            "--attention-type",
            type=str,
            metavar="EXPR",
            help="decoder attention, defaults to dot",
        )
        parser.add_argument(
            "--attention-heads",
            default=8,
            type=int,
            metavar="N",
            help="number of encoder-decoder attention heads, used when attention"
            " type is multihead. Ignored unless attention type is multihead.",
        )
        parser.add_argument(
            "--first-layer-attention",
            default=False,
            action="store_true",
            help="calculates attention after decoder's first RNN layer and"
            "concatenates it in input of every subsequent layer",
        )
        parser.add_argument(
            "--residual-level",
            default=None,
            type=int,
            help=(
                "First layer where to apply a residual connection. "
                "The value should be greater than 0 and smaller than the number of "
                "layers."
            ),
        )
        parser.add_argument(
            "--cell-type",
            default="lstm",
            type=str,
            metavar="EXPR",
            help="cell type, defaults to lstm, values:lstm, milstm, layer_norm_lstm",
        )

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument(
            "--encoder-dropout-in",
            type=float,
            metavar="D",
            help="dropout probability for encoder input embedding",
        )
        parser.add_argument(
            "--encoder-dropout-out",
            type=float,
            metavar="D",
            help="dropout probability for encoder output",
        )
        parser.add_argument(
            "--decoder-dropout-in",
            type=float,
            metavar="D",
            help="dropout probability for decoder input embedding",
        )
        parser.add_argument(
            "--decoder-dropout-out",
            type=float,
            metavar="D",
            help="dropout probability for decoder output",
        )
        parser.add_argument(
            "--sequence-lstm",
            action="store_true",
            help="use nn.LSTM implementation for encoder",
        )
        parser.add_argument(
            "--ngram-decoder",
            default=None,
            type=int,
            nargs="+",
            help=(
                "A single integer, or a list of integers. If "
                "positive, the decoder is not recurrent but a feedforward "
                "network with target-side n-gram history as input. The decoder "
                "is still conditioned on the source side via attention. If "
                "this parameter is a list of integers, the n-th entry applies "
                "to the n-th decoder (for multilingual models and "
                "multi-decoders)"
            ),
        )
        parser.add_argument(
            "--ngram-activation-type",
            default="relu",
            type=str,
            metavar="EXPR",
            help=(
                "Activation in FF layers of the ngram decoder, defaults to "
                "relu, values: relu, tanh"
            ),
        )
        parser.add_argument(
            "--multi-encoder",
            default=None,
            type=int,
            help=(
                "If this is positive, train n encoder networks rather than "
                "only one. The outputs of the encoders are concatenated before "
                "passing them through to the decoder."
            ),
        )
        parser.add_argument(
            "--multi-decoder",
            default=None,
            type=int,
            help=(
                "If this is positive, train n decoder networks rather than "
                "only one. The predictions are combined via the method in "
                "--multi-decoder-combination-strategy."
            ),
        )
        parser.add_argument(
            "--multi-decoder-combination-strategy",
            default="bottleneck",
            type=str,
            metavar="EXPR",
            help=(
                "Only used if --multi-decoder is positive. Controls how the "
                "decoders are combined with each other.\n"
                "- uniform: Separate projection layers, average predictions\n"
                "- uniform-probspace: Separate projection layers, average "
                "in probability space.\n"
                "- uniform-logprobspace: Separate projection layers, average "
                "in log-probability space.\n"
                "- unprojected: Shared projection layer, unprojected "
                "decoder outputs are averaged.\n"
                "- deepfusion: cf. https://arxiv.org/pdf/1503.03535.pdf \n"
                "- coldfusion: cf. https://arxiv.org/pdf/1708.06426.pdf \n"
                "- weighted: Separate projection layers, weighted average "
                "of logits. Weights are learned from unprojected decoder "
                "outputs.\n"
                "- weighted-probspace: Like 'weighted', but average in "
                "probability space.\n"
                "- weighted-logprobspace: Like 'weighted', but average in "
                "log-probability space.\n"
                "- weighted-unprojected: Shared projection layer, weighted "
                "average of decoder outputs. Weights are learned from "
                "unprojected decoder outputs.\n"
                "- concat: Shared projection layer, decoder outputs are "
                "concatenated.\n"
                "- bottleneck: Like 'concat' but with an additional "
                "bottleneck layer to reduce the size of the output embedding "
                "matrix.\n"
                "- deep_bottleneck: Like 'bottleneck' but with an additional "
                "non-linear layer.\n"
                "- multiplicative-unprojected: Shared projection layer, element"
                "-wise product of decoder outputs after ReLU.\n"
                "- max-unprojected: Shared projection layer, element"
                "-wise max of decoder outputs.\n"
            ),
        )
        parser.add_argument(
            "--multi-model-fixed-weights",
            default=None,
            type=float,
            nargs="+",
            help=(
                "Used for weighted* combination strategies. If specified, use "
                "these fixed model weights rather than a gating network."
            ),
        )
        parser.add_argument(
            "--multi-model-training-schedule",
            default="complete",
            type=str,
            metavar="EXPR",
            help=(
                "Only used if --multi-decoder is positive.\n"
                "- 'complete': Jointly train entire network on all batches.\n"
                "- 'unfreeze_single': Freeze all submodels except one for each "
                "training batch.\n"
                "- 'unfreeze_single_encoder': Freeze all encoders except one "
                "for each training batch.\n"
                "- 'unfreeze_single_decoder': Freeze all decoders except one "
                "for each training batch.\n"
                "- 'unfreeze_enc_N': Freeze N-th encoder.\n"
                "- 'unfreeze_dec_N': Freeze N-th decoder.\n"
                "- 'unfreeze_encdec_N': Freeze N-th encoder and N-th decoder.\n"
                "- 'freeze_all': Freeze all submodels, only train combination "
                "strategy.\n"
                "- 'freeze_all_encoders': Freeze all encoders.\n"
                "- 'freeze_all_decoders': Freeze all decoders.\n"
                "- 'separate': Each training batch is used for only one of the "
                "following: Train the n-th submodel, or train combination "
                "strategy."
            ),
        )
        parser.add_argument(
            "--multi-decoder-is-lm",
            default=None,
            type=int,
            nargs="+",
            help=(
                "If specified, sets --attention-type=no and --encoder-hidden-dim=0"
                "for the n-th decoder in an adaptive ensemble."
            ),
        )
        parser.add_argument(
            "--att-weighted-source-embeds",
            default=False,
            action="store_true",
            help=(
                "whether use attention weighted src embeddings to improve rare "
                "words translation or not"
            ),
        )
        parser.add_argument(
            "--encoder-context-embed",
            default=False,
            help=(
                "whether to use context-dependent source embeddings in the encoder "
                "for word disambiguation"
            ),
            action="store_true",
        )
        parser.add_argument(
            "--att-weighted-activation-type",
            default="tanh",
            type=str,
            metavar="EXPR",
            help=(
                "Activation in FF layers of the attention weighted src embeddings, "
                "defaults to relu, values: relu, tanh"
            ),
        )
        parser.add_argument(
            "--adaptive-softmax-cutoff",
            metavar="EXPR",
            help="comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion",
        )

        # Args for vocab reduction
        vocab_reduction.add_args(parser)

    @staticmethod
    def build_single_encoder(args, src_dict, embed_tokens):
        if args.language_model_only:
            return DummyEncoder(src_dict, num_layers=args.encoder_layers)
        if args.sequence_lstm:
            encoder_class = LSTMSequenceEncoder
        else:
            encoder_class = RNNEncoder
        return encoder_class(
            src_dict,
            embed_tokens=embed_tokens,
            embed_dim=args.encoder_embed_dim,
            cell_type=args.cell_type,
            num_layers=args.encoder_layers,
            hidden_dim=args.encoder_hidden_dim,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            residual_level=args.residual_level,
            bidirectional=bool(args.encoder_bidirectional),
            left_pad=args.left_pad_source,
            encoder_context_embed=args.encoder_context_embed,
        )

    @staticmethod
    def build_single_decoder(
        args,
        src_dict,
        dst_dict,
        embed_tokens,
        ngram_decoder=None,
        project_output=True,
        is_lm=False,
    ):
        if args.adaptive_softmax_cutoff is not None:
            project_output = False
        attention_type = args.attention_type
        encoder_hidden_dim = args.encoder_hidden_dim
        if is_lm:
            attention_type = "no"
            encoder_hidden_dim = 0
        if ngram_decoder:
            if args.ngram_activation_type == "relu":
                activation_fn = nn.ReLU
            elif args.ngram_activation_type == "tanh":
                activation_fn = nn.Tanh
            else:
                raise Exception(
                    "ngram_activation_type '%s' not implemented"
                    % args.ngram_activation_type
                )
            decoder = NGramDecoder(
                src_dict=src_dict,
                dst_dict=dst_dict,
                n=ngram_decoder,
                encoder_hidden_dim=encoder_hidden_dim,
                embed_dim=args.decoder_embed_dim,
                freeze_embed=args.decoder_freeze_embed,
                out_embed_dim=args.decoder_out_embed_dim,
                num_layers=args.decoder_layers,
                hidden_dim=args.decoder_hidden_dim,
                attention_type=attention_type,
                dropout_in=args.decoder_dropout_in,
                dropout_out=args.decoder_dropout_out,
                residual_level=args.residual_level,
                activation_fn=activation_fn,
                project_output=project_output,
                pretrained_embed=args.decoder_pretrained_embed,
                projection_pretrained_embed=args.decoder_out_pretrained_embed,
            )
        else:
            decoder = RNNDecoder(
                src_dict=src_dict,
                dst_dict=dst_dict,
                embed_tokens=embed_tokens,
                vocab_reduction_params=args.vocab_reduction_params,
                encoder_hidden_dim=encoder_hidden_dim,
                embed_dim=args.decoder_embed_dim,
                out_embed_dim=args.decoder_out_embed_dim,
                cell_type=args.cell_type,
                num_layers=args.decoder_layers,
                hidden_dim=args.decoder_hidden_dim,
                attention_type=attention_type,
                attention_heads=args.attention_heads,
                first_layer_attention=bool(args.first_layer_attention),
                dropout_in=args.decoder_dropout_in,
                dropout_out=args.decoder_dropout_out,
                residual_level=args.residual_level,
                averaging_encoder=args.averaging_encoder,
                project_output=project_output,
                pretrained_embed=args.decoder_pretrained_embed,
                projection_pretrained_embed=args.decoder_out_pretrained_embed,
                out_embed_norm=args.out_embed_norm,
                tie_embeddings=args.decoder_tie_embeddings,
                att_weighted_src_embeds=args.att_weighted_src_embeds,
                src_embed_dim=args.encoder_embed_dim,
                att_weighted_activation_type=args.att_weighted_activation_type,
                fp16=args.fp16,
            )

        # Being able to use adaptive softmax for RNN decoder
        decoder.adaptive_softmax = None

        if args.adaptive_softmax_cutoff is not None:
            decoder.adaptive_softmax = AdaptiveSoftmax(
                len(dst_dict),
                args.decoder_out_embed_dim or args.decoder_hidden_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.dropout,
            )
        return decoder

    @classmethod
    def build_embed_tokens(cls, args, src_dict, dst_dict):
        """Builds encoder and decoder token embeddings. If pretrained embeddings
        are specified, load them."""
        encoder_embed_tokens = Embedding(
            num_embeddings=len(src_dict),
            embedding_dim=args.encoder_embed_dim,
            padding_idx=src_dict.pad(),
            freeze_embed=args.encoder_freeze_embed,
            normalize_embed=args.encoder_normalize_embed,
        )
        pytorch_translate_utils.load_embedding(
            embedding=encoder_embed_tokens,
            dictionary=src_dict,
            pretrained_embed=args.encoder_pretrained_embed,
        )
        decoder_embed_tokens = Embedding(
            num_embeddings=len(dst_dict),
            embedding_dim=args.decoder_embed_dim,
            padding_idx=dst_dict.pad(),
            freeze_embed=args.decoder_freeze_embed,
        )

        pytorch_translate_utils.load_embedding(
            embedding=decoder_embed_tokens,
            dictionary=dst_dict,
            pretrained_embed=args.decoder_pretrained_embed,
        )
        return encoder_embed_tokens, decoder_embed_tokens

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        """Build a new (multi-)encoder instance."""
        if args.multi_encoder is not None:
            encoders = [
                RNNModel.build_single_encoder(args, src_dict, embed_tokens=embed_tokens)
                for _ in range(args.multi_encoder)
            ]
            encoder = MultiEncoder(
                src_dict, encoders, training_schedule=args.multi_model_training_schedule
            )
        else:
            encoder = RNNModel.build_single_encoder(
                args, src_dict, embed_tokens=embed_tokens
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, src_dict, dst_dict, embed_tokens):
        """Build a new (multi-)decoder instance."""
        if args.multi_decoder is not None:
            ngram_decoder_args = [None] * args.multi_decoder
            if args.ngram_decoder is not None:
                ngram_decoder_args = args.ngram_decoder
                if len(ngram_decoder_args) == 1:
                    ngram_decoder_args = [ngram_decoder_args[0]] * args.multi_decoder
                assert len(ngram_decoder_args) == args.multi_decoder
            is_lm_args = [False] * args.multi_decoder
            if args.multi_decoder_is_lm is not None:
                is_lm_args = list(map(bool, args.multi_decoder_is_lm))
            assert len(is_lm_args) == args.multi_decoder
            decoders = [
                RNNModel.build_single_decoder(
                    args,
                    src_dict,
                    dst_dict,
                    embed_tokens=embed_tokens,
                    ngram_decoder=n,
                    project_output=False,
                    is_lm=is_lm,
                )
                for is_lm, n in zip(is_lm_args, ngram_decoder_args)
            ]
            decoder = MultiDecoder(
                src_dict,
                dst_dict,
                decoders=decoders,
                combination_strategy=args.multi_decoder_combination_strategy,
                is_lm=is_lm_args,
                split_encoder=args.multi_encoder is not None,
                vocab_reduction_params=args.vocab_reduction_params,
                training_schedule=args.multi_model_training_schedule,
                fixed_weights=args.multi_model_fixed_weights,
            )
        else:
            if args.multi_encoder:
                args.encoder_hidden_dim *= args.multi_encoder
            n = args.ngram_decoder[0] if args.ngram_decoder else None
            decoder = RNNModel.build_single_decoder(
                args, src_dict, dst_dict, embed_tokens=embed_tokens, ngram_decoder=n
            )
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        # set default value for old checkpoints
        args.left_pad_source = getattr(args, "left_pad_source", False)
        if pytorch_translate_data.is_multilingual(args):
            return RNNModel.build_model_multilingual(args, task)
        src_dict, dst_dict = task.source_dictionary, task.target_dictionary

        encoder_embed_tokens, decoder_embed_tokens = RNNModel.build_embed_tokens(
            args, src_dict, dst_dict
        )
        encoder = RNNModel.build_encoder(
            args, src_dict, embed_tokens=encoder_embed_tokens
        )
        decoder = RNNModel.build_decoder(
            args, src_dict, dst_dict, embed_tokens=decoder_embed_tokens
        )
        return cls(task, encoder, decoder)

    @classmethod
    def build_model_multilingual(cls, args, task):
        """Build a new multilingual model instance."""
        encoders = []
        for lang in args.multiling_encoder_lang:
            d = task.source_dictionaries.get(lang, None)
            if d is not None:
                encoder_embed_tokens = Embedding(
                    num_embeddings=len(d),
                    embedding_dim=args.encoder_embed_dim,
                    padding_idx=d.pad(),
                    freeze_embed=args.encoder_freeze_embed,
                    normalize_embed=args.encoder_normalize_embed,
                )
                pytorch_translate_utils.load_embedding(
                    embedding=encoder_embed_tokens,
                    dictionary=d,
                    pretrained_embed=args.encoder_pretrained_embed,
                )
                encoders.append(
                    RNNModel.build_encoder(args, d, embed_tokens=encoder_embed_tokens)
                )
            else:
                encoders.append(None)
        encoder = MultilingualEncoder(
            task.source_dictionary,
            encoders,
            hidden_dim=args.encoder_hidden_dim,
            num_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            rescale_grads=args.multiling_rescale_grads,
        )
        decoders = []
        for lang in args.multiling_decoder_lang:
            d = task.target_dictionaries.get(lang, None)
            if d is not None:
                decoder_embed_tokens = Embedding(
                    num_embeddings=len(d),
                    embedding_dim=args.decoder_embed_dim,
                    padding_idx=d.pad(),
                    freeze_embed=args.decoder_freeze_embed,
                )

                pytorch_translate_utils.load_embedding(
                    embedding=decoder_embed_tokens,
                    dictionary=d,
                    pretrained_embed=args.decoder_pretrained_embed,
                )
                decoders.append(
                    RNNModel.build_decoder(
                        args, None, d, embed_tokens=decoder_embed_tokens
                    )
                )
            else:
                decoders.append(None)
        decoder = MultilingualDecoder(
            task.target_dictionary,
            decoders,
            hidden_dim=args.encoder_hidden_dim,
            rescale_grads=args.multiling_rescale_grads,
        )
        return cls(task, encoder, decoder)

    def get_targets(self, sample, net_output):
        targets = sample["target"].view(-1)
        possible_translation_tokens = net_output[-1]
        if possible_translation_tokens is not None:
            # pad_mask is used to ensure padding IDs remain the same regardless
            # of where torch_find() finds it in possible_translation_tokens.
            # This is important since FairseqCriterion has special logic to
            # ignore padding IDs when calculating loss.
            pad_mask = targets == self.decoder.padding_idx
            targets = torch_find(
                possible_translation_tokens, targets, len(self.task.target_dictionary)
            )
            targets[pad_mask] = self.decoder.padding_idx
        return targets


class LSTMSequenceEncoder(FairseqEncoder):
    """RNN encoder using nn.LSTM for cuDNN support / ONNX exportability."""

    @staticmethod
    def LSTM(input_size, hidden_size, **kwargs):
        m = nn.LSTM(input_size, hidden_size, **kwargs)
        for name, param in m.named_parameters():
            if "weight" in name or "bias" in name:
                param.data.uniform_(-0.1, 0.1)
        return m

    def __init__(
        self,
        dictionary,
        embed_tokens,
        embed_dim=512,
        cell_type="lstm",
        hidden_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        residual_level=None,
        bidirectional=False,
        left_pad=True,
        encoder_context_embed=False,
    ):
        assert cell_type == "lstm", 'sequence-lstm requires cell_type="lstm"'

        super().__init__(dictionary)
        self.dictionary = dictionary
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.residual_level = residual_level
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.padding_idx = dictionary.pad()
        self.left_pad = left_pad
        self.embed_tokens = embed_tokens

        if encoder_context_embed:
            self.embed_tokens_context = ContextEmbedding(embed_dim=embed_dim)
        self.encoder_context_embed = encoder_context_embed

        self.word_dim = embed_dim

        self.bilstm = BiLSTM(
            num_layers=num_layers,
            bidirectional=bidirectional,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout_out,
            residual_level=residual_level,
        )

        # Variable tracker
        self.tracker = VariableTracker()

        # Initialize adversarial mode
        self.set_gradient_tracking_mode(False)
        self.set_embed_noising_mode(False)

    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens, self.padding_idx, left_to_right=True
            )

        # If we're generating adversarial examples we need to keep track of
        # some internal variables
        self.tracker.reset()

        bsz, seqlen = pytorch_translate_utils.get_source_tokens_tensor(
            src_tokens
        ).size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        if self.encoder_context_embed:
            x = self.embed_tokens_context(x)

        # Apply feature level noising is specified
        if self.embed_noising_mode and self.embed_noising_layer is not None:
            x = self.embed_noising_layer(x)
        # Track token embeddings for generation white-box adversarial example
        self.tracker.track(x, "token_embeddings", retain_grad=self.track_gradients)

        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        embedded_words = x

        # Allows compatibility with Caffe2 inputs for tracing (int32)
        # as well as the current format of Fairseq-Py inputs (int64)
        if src_lengths.dtype is torch.int64:
            src_lengths = src_lengths.int()

        unpacked_output, final_hiddens, final_cells = self.bilstm(
            embeddings=x, lengths=src_lengths
        )

        return (
            unpacked_output,
            final_hiddens,
            final_cells,
            src_lengths,
            src_tokens,
            embedded_words,
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder all outputs according to new_order."""
        return reorder_encoder_output(encoder_out, new_order)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number

    def set_gradient_tracking_mode(self, mode=True):
        self.tracker.reset()
        self.track_gradients = mode

    def set_embed_noising_mode(self, mode=True):
        """This allows adversarial trainer to turn on and off embedding noising
        layers. In regular training, this mode is off, and it is not included
        in forward pass.
        """
        self.embed_noising_mode = mode


class DummyEncoder(FairseqEncoder):
    """Dummy encoder which outputs None. Used for LM training."""

    def __init__(self, dictionary, num_layers=1):
        super().__init__(dictionary)
        self.num_layers = num_layers

    def forward(self, src_tokens, src_lengths):
        bsz = src_lengths.size(0)
        ones = maybe_cuda(torch.ones((self.num_layers, bsz, 1)))
        dummy_out = maybe_cuda(torch.ones((1, bsz, 1)))

        # ones are returned so that FC layer corresponds to learned initial
        # state for language model
        return dummy_out, ones, ones, src_lengths, src_tokens, dummy_out

    def reorder_encoder_out(self, encoder_out, new_order):
        return reorder_encoder_output(encoder_out, new_order)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class RNNEncoder(FairseqEncoder):
    """RNN encoder."""

    def __init__(
        self,
        dictionary,
        embed_tokens,
        embed_dim=512,
        hidden_dim=512,
        num_layers=1,
        cell_type="lstm",
        dropout_in=0.1,
        dropout_out=0.1,
        residual_level=None,
        bidirectional=False,
        left_pad=True,
        encoder_context_embed=False,
    ):
        super().__init__(dictionary)
        self.dictionary = dictionary
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.residual_level = residual_level
        self.hidden_dim = hidden_dim
        self.output_units = hidden_dim  # fairseq LSTM compatibility
        self.bidirectional = bidirectional
        self.padding_idx = dictionary.pad()
        self.left_pad = left_pad
        self.embed_tokens = embed_tokens

        if encoder_context_embed:
            self.embed_tokens = ContextEmbedding(
                embed_dim=self.embed_dim
            ).encoder_embed_fn
        self.encoder_context_embed = encoder_context_embed
        self.word_dim = embed_dim

        self.cell_type = cell_type
        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            self.layers.append(
                RNNLayer(
                    self.word_dim if layer == 0 else hidden_dim,
                    hidden_dim,
                    self.cell_type,
                    True if bidirectional and layer == 0 else False,
                )
            )

        self.num_layers = len(self.layers)

    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens, self.padding_idx, left_to_right=True
            )
        bsz, seqlen = pytorch_translate_utils.get_source_tokens_tensor(
            src_tokens
        ).size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        if self.encoder_context_embed:
            x = self.embed_tokens_context(x)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        embedded_words = x

        # Generate packed seq to deal with varying source seq length
        packed_input, batch_sizes, _, _ = pack_padded_sequence(x, src_lengths)
        final_hiddens, final_cells = [], []
        next_hiddens = []
        for i, rnn_layer in enumerate(self.layers):
            current_hidden_size = (
                self.hidden_dim // 2 if rnn_layer.is_bidirectional else self.hidden_dim
            )

            if self.cell_type in ["lstm", "milstm", "layer_norm_lstm"]:
                prev_hidden = (
                    x.new(bsz, current_hidden_size).zero_(),
                    x.new(bsz, current_hidden_size).zero_(),
                )
            else:
                raise Exception(f"{self.cell_type} not implemented")

            hidden, current_output = rnn_layer.forward(
                packed_input, prev_hidden, batch_sizes
            )
            next_hiddens.append(hidden)
            prev_hidden = next_hiddens[-1]

            if self.dropout_out != 0:
                current_output = F.dropout(
                    current_output, p=self.dropout_out, training=self.training
                )

            if self.residual_level is not None and i >= self.residual_level:
                packed_input = packed_input.clone() + current_output
            else:
                packed_input = current_output

        final_hiddens, final_cells = zip(*next_hiddens)
        # Reshape to [num_layer, batch_size, hidden_dim]
        final_hiddens = torch.cat(final_hiddens, dim=0).view(
            self.num_layers, *final_hiddens[0].size()
        )
        final_cells = torch.cat(final_cells, dim=0).view(
            self.num_layers, *final_cells[0].size()
        )

        #  [max_seqlen, batch_size, hidden_dim]
        unpacked_output, _ = pad_packed_sequence(
            PackedSequence(packed_input, batch_sizes)
        )

        return (
            unpacked_output,
            final_hiddens,
            final_cells,
            src_lengths,
            src_tokens,
            embedded_words,
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder all outputs according to new_order."""
        return reorder_encoder_output(encoder_out, new_order)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class RNNDecoder(DecoderWithOutputProjection):
    """
    RNN decoder with multihead attention. Attention is calculated using encoder
    output and output of decoder's first RNN layerself. Attention is applied
    after first RNN layer and concatenated to input of subsequent layers.
    """

    def __init__(
        self,
        src_dict,
        dst_dict,
        embed_tokens,
        vocab_reduction_params=None,
        encoder_hidden_dim=512,
        embed_dim=512,
        hidden_dim=512,
        out_embed_dim=512,
        cell_type="lstm",
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        attention_type="dot",
        attention_heads=8,
        first_layer_attention=False,
        residual_level=None,
        averaging_encoder=False,
        project_output=True,
        out_embed_norm=None,
        tie_embeddings=False,
        pretrained_embed=None,
        projection_pretrained_embed=None,
        att_weighted_src_embeds=False,
        src_embed_dim=512,
        att_weighted_activation_type="tanh",
        predictor=None,
        fp16: bool = False,
    ):
        super().__init__(
            src_dict,
            dst_dict,
            vocab_reduction_params,
            out_embed_dim,
            project_output=project_output,
            pretrained_embed=projection_pretrained_embed,
            out_embed_norm=out_embed_norm,
            att_weighted_src_embeds=att_weighted_src_embeds,
            src_embed_dim=src_embed_dim,
            att_weighted_activation_type=att_weighted_activation_type,
            predictor=predictor,
            fp16=fp16,
        )
        encoder_hidden_dim = max(1, encoder_hidden_dim)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_embed_dim = out_embed_dim
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.attention_type = attention_type
        self.residual_level = residual_level
        self.tie_embeddings = tie_embeddings
        self.attention_heads = attention_heads
        self.first_layer_attention = first_layer_attention
        self.padding_idx = dst_dict.pad()
        self.embed_tokens = embed_tokens

        if self.tie_embeddings:
            assert self.embed_dim == self.out_embed_dim, (
                "Input embeddings and output projections must have the same "
                "dimension for the weights to be tied"
            )
            self.embed_tokens.weight = self.output_projection_w

        self.hidden_dim = hidden_dim
        self.averaging_encoder = averaging_encoder

        if cell_type == "lstm":
            cell_class = rnn_cell.LSTMCell
        elif cell_type == "milstm":
            cell_class = rnn_cell.MILSTMCell
        elif cell_type == "layer_norm_lstm":
            cell_class = rnn_cell.LayerNormLSTMCell

        if hidden_dim != encoder_hidden_dim:
            hidden_init_fc_list = []
            cell_init_fc_list = []
            for _ in range(num_layers):
                hidden_init_fc_list.append(Linear(encoder_hidden_dim, hidden_dim))
                cell_init_fc_list.append(Linear(encoder_hidden_dim, hidden_dim))
            self.hidden_init_fc_list = nn.ModuleList(hidden_init_fc_list)
            self.cell_init_fc_list = nn.ModuleList(cell_init_fc_list)

        self.attention = attention.build_attention(
            attention_type=attention_type,
            decoder_hidden_state_dim=hidden_dim,
            context_dim=encoder_hidden_dim,
            nheads=attention_heads,  # specific to multihead_attention
        )

        if self.attention.context_dim:
            self.initial_attn_context = nn.Parameter(
                torch.Tensor(self.attention.context_dim).zero_()
            )
        self.combined_output_and_context_dim = self.attention.context_dim + hidden_dim

        layers = []
        for layer in range(num_layers):
            if layer == 0:
                cell_input_dim = embed_dim
            else:
                cell_input_dim = hidden_dim

            # attention applied to first layer always.
            if self.first_layer_attention or layer == 0:
                cell_input_dim += self.attention.context_dim
            layers.append(cell_class(input_dim=cell_input_dim, hidden_dim=hidden_dim))
        self.layers = nn.ModuleList(layers)

        if self.combined_output_and_context_dim != out_embed_dim:
            self.additional_fc = Linear(
                self.combined_output_and_context_dim, out_embed_dim
            )

    def forward_unprojected(self, input_tokens, encoder_out, incremental_state=None):
        if incremental_state is not None:
            input_tokens = input_tokens[:, -1:]
        bsz, seqlen = input_tokens.size()

        # get outputs from encoder
        (
            encoder_outs,
            final_hidden,
            final_cell,
            src_lengths,
            src_tokens,
            _,
        ) = encoder_out

        # embed tokens
        x = self.embed_tokens(input_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(
            self, incremental_state, "cached_state"
        )
        input_feed = None
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            # first time step, initialize previous states
            init_prev_states = self._init_prev_states(encoder_out)
            prev_hiddens = []
            prev_cells = []

            # init_prev_states may or may not include initial attention context
            for (h, c) in zip(init_prev_states[0::2], init_prev_states[1::2]):
                prev_hiddens.append(h)
                prev_cells.append(c)
            if self.attention.context_dim:
                input_feed = self.initial_attn_context.expand(
                    bsz, self.attention.context_dim
                )

        attn_scores_per_step = []
        outs = []
        step_attn_scores = None
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            step_input = maybe_cat((x[j, :, :], input_feed), dim=1)
            previous_layer_input = step_input
            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(step_input, (prev_hiddens[i], prev_cells[i]))

                if self.first_layer_attention and i == 0:
                    # tgt_len is 1 in decoder and squeezed for both matrices
                    # input_feed.shape = tgt_len X bsz X embed_dim
                    # step_attn_scores.shape = src_len X tgt_len X bsz
                    input_feed, step_attn_scores = self.attention(
                        hidden, encoder_outs, src_lengths
                    )

                # hidden state becomes the input to the next layer
                layer_output = F.dropout(
                    hidden, p=self.dropout_out, training=self.training
                )

                if self.residual_level is not None and i >= self.residual_level:
                    # TODO add an assert related to sizes here
                    step_input = layer_output + previous_layer_input
                else:
                    step_input = layer_output

                if self.first_layer_attention:
                    step_input = maybe_cat((step_input, input_feed), dim=1)
                previous_layer_input = step_input

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            if not self.first_layer_attention:
                input_feed, step_attn_scores = self.attention(
                    hidden, encoder_outs, src_lengths
                )

            attn_scores_per_step.append(step_attn_scores.unsqueeze(1))
            attn_scores = torch.cat(attn_scores_per_step, dim=1)
            # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
            attn_scores = attn_scores.transpose(0, 2)
            combined_output_and_context = maybe_cat((hidden, input_feed), dim=1)
            # save final output
            outs.append(combined_output_and_context)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self,
            incremental_state,
            "cached_state",
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(
            seqlen, bsz, self.combined_output_and_context_dim
        )

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # bottleneck layer
        if hasattr(self, "additional_fc"):
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
        return x, attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        cached_state = utils.get_incremental_state(
            self, incremental_state, "cached_state"
        )
        if cached_state is None:
            return

        def reorder_state(state):
            if state is None:
                return None
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, "cached_state", new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def _init_prev_states(self, encoder_out):
        (
            encoder_output,
            final_hiddens,
            final_cells,
            src_lengths,
            src_tokens,
            _,
        ) = encoder_out
        num_layers = len(self.layers)
        if self.averaging_encoder:
            # Use mean encoder hidden states
            prev_hiddens = [torch.mean(encoder_output, 0)] * num_layers
        else:
            # Simply return the final state of each layer
            prev_hiddens = [final_hiddens[i] for i in range(num_layers)]
        prev_cells = [final_cells[i] for i in range(num_layers)]

        if hasattr(self, "hidden_init_fc_list"):
            for i in range(num_layers):
                prev_hiddens[i] = self.hidden_init_fc_list[i](prev_hiddens[i])
                prev_cells[i] = self.cell_init_fc_list[i](prev_cells[i])

        prev_states = []
        for h, c in zip(prev_hiddens, prev_cells):
            prev_states.extend([h, c])
        if self.attention.context_dim:
            prev_states.append(self.initial_attn_context)

        return prev_states


class BiLSTM(nn.Module):
    """Wrapper for nn.LSTM

    Differences include:
    * weight initialization
    * the bidirectional option makes the first layer bidirectional only
    (and in that case the hidden dim is divided by 2)
    """

    @staticmethod
    def LSTM(input_size, hidden_size, **kwargs):
        m = nn.LSTM(input_size, hidden_size, **kwargs)
        for name, param in m.named_parameters():
            if "weight" in name or "bias" in name:
                param.data.uniform_(-0.1, 0.1)
        return m

    def __init__(
        self, num_layers, bidirectional, embed_dim, hidden_dim, dropout, residual_level
    ):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if bidirectional:
            assert hidden_dim % 2 == 0, "hidden_dim should be even if bidirectional"
        self.hidden_dim = hidden_dim
        self.residual_level = residual_level
        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            is_layer_bidirectional = bidirectional and layer == 0
            if is_layer_bidirectional:
                assert hidden_dim % 2 == 0, (
                    "hidden_dim must be even if bidirectional "
                    "(to be divided evenly between directions)"
                )
            self.layers.append(
                BiLSTM.LSTM(
                    embed_dim if layer == 0 else hidden_dim,
                    hidden_dim // 2 if is_layer_bidirectional else hidden_dim,
                    num_layers=1,
                    dropout=dropout,
                    bidirectional=is_layer_bidirectional,
                )
            )

    def forward(self, embeddings, lengths, enforce_sorted=True):
        # enforce_sorted is set to True by default to force input lengths
        # are sorted in a descending order when pack padded sequence.
        bsz = embeddings.size()[1]

        # Generate packed seq to deal with varying source seq length
        # packed_input is of type PackedSequence, which consists of:
        # element [0]: a tensor, the packed data, and
        # element [1]: a list of integers, the batch size for each step
        packed_input = pack_padded_sequence(
            embeddings, lengths, enforce_sorted=enforce_sorted
        )

        final_hiddens, final_cells = [], []
        for i, rnn_layer in enumerate(self.layers):
            if self.bidirectional and i == 0:
                h0 = embeddings.new(2, bsz, self.hidden_dim // 2).zero_()
                c0 = embeddings.new(2, bsz, self.hidden_dim // 2).zero_()
            else:
                h0 = embeddings.new(1, bsz, self.hidden_dim).zero_()
                c0 = embeddings.new(1, bsz, self.hidden_dim).zero_()

            # apply LSTM along entire sequence
            current_output, (h_last, c_last) = rnn_layer(packed_input, (h0, c0))

            # final state shapes: (bsz, hidden_dim)
            if self.bidirectional and i == 0:
                # concatenate last states for forward and backward LSTM
                h_last = torch.cat((h_last[0, :, :], h_last[1, :, :]), dim=1)
                c_last = torch.cat((c_last[0, :, :], c_last[1, :, :]), dim=1)
            else:
                h_last = h_last.squeeze(dim=0)
                c_last = c_last.squeeze(dim=0)

            final_hiddens.append(h_last)
            final_cells.append(c_last)

            if self.residual_level is not None and i >= self.residual_level:
                packed_input[0] = packed_input.clone()[0] + current_output[0]
            else:
                packed_input = current_output

        # Reshape to [num_layer, batch_size, hidden_dim]
        final_hiddens = torch.cat(final_hiddens, dim=0).view(
            self.num_layers, *final_hiddens[0].size()
        )
        final_cells = torch.cat(final_cells, dim=0).view(
            self.num_layers, *final_cells[0].size()
        )

        #  [max_seqlen, batch_size, hidden_dim]
        unpacked_output, _ = pad_packed_sequence(packed_input)

        return (unpacked_output, final_hiddens, final_cells)


@register_model("semi_supervised_rnn")
class SemisupervisedRNNModel(SemiSupervisedModel):
    single_model_cls = RNNModel

    @staticmethod
    def add_args(parser):
        RNNModel.add_args(parser)
        SemiSupervisedModel.add_args(parser)


@register_model_architecture("rnn", "rnn")
def base_architecture(args):
    # default architecture
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 1)
    args.encoder_hidden_dim = getattr(args, "encoder_hidden_dim", 512)
    args.encoder_bidirectional = getattr(args, "encoder_bidirectional", False)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", args.dropout)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", args.dropout)
    args.encoder_pretrained_embed = getattr(args, "encoder_pretrained_embed", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_hidden_dim = getattr(args, "decoder_hidden_dim", 512)
    args.decoder_pretrained_embed = getattr(args, "decoder_pretrained_embed", None)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_out_pretrained_embed = getattr(
        args, "decoder_out_pretrained_embed", None
    )
    args.attention_type = getattr(args, "attention_type", "dot")
    args.attention_heads = getattr(args, "attention_heads", 8)
    args.first_layer_attention = getattr(args, "first_layer_attention", False)
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.averaging_encoder = getattr(args, "averaging_encoder", False)
    args.encoder_freeze_embed = getattr(args, "encoder_freeze_embed", False)
    args.encoder_normalize_embed = getattr(args, "encoder_normalize_embed", False)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.ngram_decoder = getattr(args, "ngram_decoder", None)
    args.multi_encoder = getattr(args, "multi_encoder", None)
    args.multi_decoder = getattr(args, "multi_decoder", None)
    args.multi_decoder_is_lm = getattr(args, "multi_decoder_is_lm", None)
    args.multiling_encoder_lang = getattr(args, "multiling_encoder_lang", None)
    args.multi_model_training_schedule = getattr(
        args, "multi_model_training_schedule", "complete"
    )
    args.multi_model_fixed_weights = getattr(args, "multi_model_fixed_weights", None)
    args.cell_type = getattr(args, "cell_type", "lstm")
    args.ngram_activation_type = getattr(args, "ngram_activation_type", "relu")
    vocab_reduction.set_arg_defaults(args)
    args.sequence_lstm = getattr(args, "sequence_lstm", False)
    args.decoder_tie_embeddings = getattr(args, "decoder_tie_embeddings", False)
    args.encoder_pretrained_embed = getattr(args, "encoder_pretrained_embed", None)
    args.decoder_pretrained_embed = getattr(args, "decoder_pretrained_embed", None)
    args.decoder_out_pretrained_embed = getattr(
        args, "decoder_out_pretrained_embed", None
    )
    args.out_embed_norm = getattr(args, "out_embed_dim", None)
    args.att_weighted_src_embeds = getattr(args, "att_weighted_source_embeds", False)
    args.att_weighted_activation_type = getattr(
        args, "att_weighted_activation_type", "tanh"
    )
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.language_model_only = getattr(args, "language_model_only", False)
    args.encoder_context_embed = getattr(args, "encoder_context_embed", False)


@register_model_architecture("rnn", "rnn_big_test")
def rnn_big_test(args):
    base_architecture(args)
    args.encoder_embed_dim = 1024
    args.encoder_layers = 6
    args.encoder_hidden_dim = 1024
    args.decoder_embed_dim = 1024
    args.decoder_layers = 6
    args.decoder_hidden_dim = 1024
    args.decoder_out_embed_dim = 1024


@register_model_architecture("semi_supervised_rnn", "semi_supervised_rnn")
def semi_supervised_rnn(args):
    base_architecture(args)
    SemiSupervisedModel.set_semi_supervised_arch_args(args)
