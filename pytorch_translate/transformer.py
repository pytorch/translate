#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqModel,
    register_model,
    register_model_architecture,
    transformer as fairseq_transformer,
)
from fairseq.modules import AdaptiveSoftmax, SinusoidalPositionalEmbedding
from pytorch_translate import utils as pytorch_translate_utils, vocab_reduction
from pytorch_translate.common_layers import (
    TransformerEmbedding,
    TransformerEncoderGivenEmbeddings,
    TransformerTokenEmbedding,
    VariableTracker,
)
from pytorch_translate.semi_supervised import SemiSupervisedModel
from pytorch_translate.utils import torch_find


def build_embedding(dictionary, embed_dim, path=None, freeze=False):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = TransformerTokenEmbedding(num_embeddings, embed_dim, padding_idx, freeze)
    # if provided, load from preloaded dictionaries
    if path:
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, emb)
    return emb


@register_model("ptt_transformer")
class TransformerModel(FairseqModel):
    def __init__(self, task, encoder, decoder):
        super().__init__(encoder, decoder)
        self.task = task

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after ReLU in FFN",
        )
        parser.add_argument(
            "--encoder-pretrained-embed",
            type=str,
            metavar="STR",
            help="path to pre-trained encoder embedding",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
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
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            default=False,
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--encoder-learned-pos",
            default=False,
            action="store_true",
            help="use learned positional embeddings in the encoder",
        )
        parser.add_argument(
            "--decoder-pretrained-embed",
            type=str,
            metavar="STR",
            help="path to pre-trained decoder embedding",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-freeze-embed",
            default=False,
            action="store_true",
            help=(
                "whether to freeze the encoder embedding or allow it to be "
                "updated during training"
            ),
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-learned-pos",
            default=False,
            action="store_true",
            help="use learned positional embeddings in the decoder",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            default=False,
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            default=False,
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--share-all-embeddings",
            default=False,
            action="store_true",
            help="share encoder, decoder and output embeddings"
            " (requires shared dictionary and embed dim)",
        )
        parser.add_argument(
            "--adaptive-softmax-cutoff",
            default=None,
            metavar="EXPR",
            help="comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion",
        )

        # Args for vocab reduction
        vocab_reduction.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        base_architecture(args)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError(
                    "--share-all-embeddings requires a joined dictionary"
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    "--share-all-embeddings requires --encoder-embed-dim "
                    "to match --decoder-embed-dim"
                )
            if args.decoder_pretrained_embed and (
                args.decoder_pretrained_embed != args.encoder_pretrained_embed
            ):
                raise RuntimeError(
                    "--share-all-embeddings not compatible with "
                    "--decoder-pretrained-embed"
                )
            encoder_embed_tokens = build_embedding(
                dictionary=src_dict,
                embed_dim=args.encoder_embed_dim,
                path=args.encoder_pretrained_embed,
                freeze=args.encoder_freeze_embed,
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                dictionary=src_dict,
                embed_dim=args.encoder_embed_dim,
                path=args.encoder_pretrained_embed,
                freeze=args.encoder_freeze_embed,
            )
            decoder_embed_tokens = build_embedding(
                dictionary=tgt_dict,
                embed_dim=args.decoder_embed_dim,
                path=args.decoder_pretrained_embed,
                freeze=args.decoder_freeze_embed,
            )

        encoder = TransformerModel.build_encoder(
            args, src_dict, embed_tokens=encoder_embed_tokens
        )
        decoder = TransformerModel.build_decoder(
            args, src_dict, tgt_dict, embed_tokens=decoder_embed_tokens
        )
        return TransformerModel(task, encoder, decoder)

    def get_targets(self, sample, net_output):
        targets = sample["target"].view(-1)
        possible_translation_tokens = net_output[-1]
        if possible_translation_tokens is not None:
            targets = torch_find(
                possible_translation_tokens, targets, len(self.task.target_dictionary)
            )
        return targets

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens=embed_tokens)

    @classmethod
    def build_decoder(cls, args, src_dict, dst_dict, embed_tokens):
        return TransformerDecoder(args, src_dict, dst_dict, embed_tokens=embed_tokens)


class TransformerEncoder(FairseqEncoder):
    """Transformer encoder."""

    def __init__(
        self, args, dictionary, embed_tokens, proj_to_decoder=True
    ):
        super().__init__(dictionary)
        self.transformer_embedding = TransformerEmbedding(
            args=args, embed_tokens=embed_tokens,
        )

        self.transformer_encoder_given_embeddings = TransformerEncoderGivenEmbeddings(
            args=args, proj_to_decoder=proj_to_decoder
        )

        # Variable tracker
        self.tracker = VariableTracker()

        # Initialize adversarial mode
        self.set_gradient_tracking_mode(False)
        self.set_embed_noising_mode(False)

    def forward(self, src_tokens, src_lengths):
        # Initialize the tracker to keep track of internal variables
        self.tracker.reset()
        x, encoder_padding_mask, positions = self.transformer_embedding(
            src_tokens=src_tokens, src_lengths=src_lengths
        )
        # Track token embeddings
        self.tracker.track(x, "token_embeddings", retain_grad=self.track_gradients)

        x = self.transformer_encoder_given_embeddings(
            x=x, positions=positions, encoder_padding_mask=encoder_padding_mask
        )

        # TODO(jamesreed): this is kinda a hack because we can't annotate an
        # Optional[Tensor] output for encoder_padding_mask
        if encoder_padding_mask is None:
            encoder_padding_mask = torch.empty([])

        return x, src_tokens, encoder_padding_mask

    def reorder_encoder_out(self, encoder_out, new_order):
        (x, src_tokens, encoder_padding_mask) = encoder_out
        src_tokens_tensor = pytorch_translate_utils.get_source_tokens_tensor(src_tokens)
        if x is not None:
            x = x.index_select(1, new_order)
        if src_tokens_tensor is not None:
            src_tokens_tensor = src_tokens_tensor.index_select(0, new_order)
        if encoder_padding_mask.shape == torch.Size([]):
            encoder_padding_mask = None
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.index_select(0, new_order)
        return (x, src_tokens_tensor, encoder_padding_mask)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.transformer_embedding.embed_positions.max_positions()

    def upgrade_state_dict_named(self, state_dict, name):
        if isinstance(
            self.transformer_embedding.embed_positions, SinusoidalPositionalEmbedding
        ):
            if f"{name}.transformer_embedding.embed_positions.weights" in state_dict:
                del state_dict[f"{name}.transformer_embedding.embed_positions.weights"]
            state_dict[
                f"{name}.transformer_embedding.embed_positions._float_tensor"
            ] = torch.FloatTensor(1)
        self.transformer_encoder_given_embeddings.upgrade_state_dict_named(
            state_dict, f"{name}.transformer_encoder_given_embeddings"
        )
        return state_dict

    def set_gradient_tracking_mode(self, mode=True):
        self.tracker.reset()
        self.track_gradients = mode

    def set_embed_noising_mode(self, mode=True):
        """This allows adversarial trainer to turn on and off embedding noising
        layers. In regular training, this mode is off, and it is not included
        in forward pass.
        """
        self.embed_noising_mode = mode


class TransformerDecoder(FairseqIncrementalDecoder):
    """Transformer decoder."""

    def __init__(self, args, src_dict, dst_dict, embed_tokens):
        super().__init__(dst_dict)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = fairseq_transformer.PositionalEmbedding(
            1024,
            embed_dim,
            padding_idx,
            learned=args.decoder_learned_pos,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                fairseq_transformer.TransformerDecoderLayer(args)
                for i in range(args.decoder_layers)
            ]
        )

        self.adaptive_softmax = None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dst_dict),
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.dropout,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dst_dict), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

        self.vocab_reduction_module = None
        if args.vocab_reduction_params:
            assert (
                self.adaptive_softmax is None
            ), "vocabulary reduction not compatible with adaptive softmax!"
            self.vocab_reduction_module = vocab_reduction.VocabReduction(
                src_dict, dst_dict, args.vocab_reduction_params, fp16=args.fp16
            )

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state=None,
        possible_translation_tokens=None,
        timestep=None,
    ):
        (encoder_x, src_tokens, encoder_padding_mask) = encoder_out

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens, incremental_state=incremental_state, timestep=timestep
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        state_outputs = []  # onnx_trace only
        for i, layer in enumerate(self.layers):
            if self.onnx_trace:
                # (prev_key, prev_value)
                self_attn_input = incremental_state[4 * i : 4 * i + 2]
                attn_state = incremental_state[4 * i + 2 : 4 * i + 4]
                x, attn, self_attn_out = layer(
                    x,
                    encoder_x,
                    encoder_padding_mask,
                    incremental_state={},
                    prev_self_attn_state=self_attn_input,
                    prev_attn_state=attn_state,
                )
                state_outputs.extend(self_attn_out)
                state_outputs.extend(attn_state)  # unchanged
            else:
                x, attn = layer(
                    x,
                    encoder_x,
                    encoder_padding_mask,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x)
                    if incremental_state is None
                    else None,
                )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.adaptive_softmax is not None:
            return x, attn, None

        # project back to size of vocabulary
        if self.share_input_output_embed:
            output_weights = self.embed_tokens.weight
        else:
            output_weights = self.embed_out

        if (
            self.vocab_reduction_module is not None
            and possible_translation_tokens is None
        ):
            decoder_input_tokens = prev_output_tokens.contiguous()
            possible_translation_tokens = self.vocab_reduction_module(
                src_tokens, decoder_input_tokens=decoder_input_tokens
            )
        if possible_translation_tokens is not None:
            output_weights = output_weights.index_select(
                dim=0, index=possible_translation_tokens
            )

        logits = F.linear(x, output_weights)

        if self.onnx_trace:
            return logits, attn, possible_translation_tokens, state_outputs

        return logits, attn, possible_translation_tokens

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if f"{name}.embed_positions.weights" in state_dict:
                del state_dict[f"{name}.embed_positions.weights"]
            state_dict[f"{name}.embed_positions._float_tensor"] = torch.FloatTensor(1)
        return state_dict

    def _init_prev_states(self, encoder_out):
        """
        For self-attention, initial (prev_key, prev_value) are dummy tensors
        with a zero-size sequence dimension.
        For encoder-decoder attention, key and value are computed once from
        the encoder outputs and stay the same throughout decoding.
        """
        encoder_x, src_tokens, encoder_padding_mask = encoder_out
        batch_size = torch.onnx.operators.shape_as_tensor(encoder_x)[1]
        states = []
        for layer in self.layers:
            # dummy initial (prev_key, prev_value) for self-attention
            for _ in range(2):
                dummy_state_shape = torch.cat(
                    [
                        batch_size.view(1),
                        torch.LongTensor([layer.self_attn.num_heads]),
                        torch.LongTensor([0]),
                        torch.LongTensor([layer.self_attn.head_dim]),
                    ]
                )
                dummy_state = torch.zeros(
                    [1, layer.self_attn.num_heads, 0, layer.self_attn.head_dim]
                )
                reshaped_dummy_state = torch.onnx.operators.reshape_from_tensor_shape(
                    dummy_state, dummy_state_shape
                )
                states.append(reshaped_dummy_state)

            # (key, value) for encoder-decoder attention computed from encoder
            # output and remain the same throughout decoding
            key = layer.encoder_attn.in_proj_k(encoder_x)
            value = layer.encoder_attn.in_proj_v(encoder_x)

            # (key, value) kept in shape (bsz, num_heads, seq_len, head_dim)
            # to avoid repeated transpose operations
            seq_len, batch_size_int, _ = encoder_x.shape
            num_heads = layer.encoder_attn.num_heads
            head_dim = layer.encoder_attn.head_dim
            key = (
                key.view(seq_len, batch_size_int * num_heads, head_dim)
                .transpose(0, 1)
                .view(batch_size_int, num_heads, seq_len, head_dim)
            )
            value = (
                value.view(seq_len, batch_size_int * num_heads, head_dim)
                .transpose(0, 1)
                .view(batch_size_int, num_heads, seq_len, head_dim)
            )

            states.extend([key, value])

        return states


@register_model("semi_supervised_transformer")
class SemiSupervisedTransformerModel(SemiSupervisedModel):
    """
    We can't use `self.single_model_cls` because at this point `__init__` hasn't
    run. single_model_cls is a static class variable that is meant to be
    constant
    """

    single_model_cls = TransformerModel

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        SemiSupervisedModel.add_args(parser)


@register_model_architecture("ptt_transformer", "ptt_transformer")
def base_architecture(args):
    args.encoder_pretrained_embed = getattr(args, "encoder_pretrained_embed", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_freeze_embed = getattr(args, "encoder_freeze_embed", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_pretrained_embed = getattr(args, "decoder_pretrained_embed", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    vocab_reduction.set_arg_defaults(args)


@register_model_architecture(
    "semi_supervised_transformer", "semi_supervised_transformer"
)
def semi_supervised_transformer(args):
    base_architecture(args)
    SemiSupervisedModel.set_semi_supervised_arch_args(args)
