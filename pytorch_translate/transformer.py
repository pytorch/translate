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
from pytorch_translate.common_layers import Embedding, VariableTracker


@register_model("ptt_transformer")
class TransformerModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

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
            "--encoder-embed-path",
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
            "--decoder-embed-path",
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

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        base_architecture(args)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, freeze, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx, freeze)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

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
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise RuntimeError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = build_embedding(
                src_dict,
                args.encoder_embed_dim,
                args.encoder_embed_path,
                args.encoder_freeze_embed,
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict,
                args.encoder_embed_dim,
                args.encoder_embed_path,
                args.encoder_freeze_embed,
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict,
                args.decoder_embed_dim,
                args.decoder_embed_path,
                args.decoder_freeze_embed,
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerModel(encoder, decoder)


class TransformerEncoder(FairseqEncoder):
    """Transformer encoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = fairseq_transformer.PositionalEmbedding(
            1024,
            embed_dim,
            self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                fairseq_transformer.TransformerEncoderLayer(args)
                for i in range(args.encoder_layers)
            ]
        )

        # Variable tracker
        self.tracker = VariableTracker()

        # Initialize adversarial mode
        self.set_gradient_tracking_mode(False)

    def forward(self, src_tokens, src_lengths):
        # Initialize the tracker to keep track of internal variables
        self.tracker.reset()
        # Embed tokens
        x = self.embed_tokens(src_tokens)
        # Track token embeddings
        self.tracker.track(x, "token_embeddings", retain_grad=self.track_gradients)
        # Add position embeddings and dropout
        x = self.embed_scale * x
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask (B x T)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        return x, encoder_padding_mask

    def reorder_encoder_out(self, encoder_out, new_order):
        (x, encoder_padding_mask) = encoder_out
        if x is not None:
            x = x.index_select(1, new_order)
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.index_select(0, new_order)
        return (x, encoder_padding_mask)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if "encoder.embed_positions.weights" in state_dict:
                del state_dict["encoder.embed_positions.weights"]
            state_dict["encoder.embed_positions._float_tensor"] = torch.FloatTensor(1)
        return state_dict

    def set_gradient_tracking_mode(self, mode=True):
        self.tracker.reset()
        self.track_gradients = mode


class TransformerDecoder(FairseqIncrementalDecoder):
    """Transformer decoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
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
            left_pad=left_pad,
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
                len(dictionary),
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.dropout,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        (encoder_x, encoder_padding_mask) = encoder_out

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens, incremental_state=incremental_state
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
        for layer in self.layers:
            x, attn = layer(x, encoder_x, encoder_padding_mask, incremental_state)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, attn

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if "decoder.embed_positions.weights" in state_dict:
                del state_dict["decoder.embed_positions.weights"]
            state_dict["decoder.embed_positions._float_tensor"] = torch.FloatTensor(1)
        return state_dict


@register_model_architecture("ptt_transformer", "ptt_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_freeze_embed = getattr(args, "encoder_freeze_embed", False)
    args.encoder_learned_pos = getattr(args, "encoder_normalize_before", False)
    args.encoder_normalize_before = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.decoder_learned_pos = getattr(args, "decoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_learned_pos", False)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.)
    args.relu_dropout = getattr(args, "relu_dropout", 0.)
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
