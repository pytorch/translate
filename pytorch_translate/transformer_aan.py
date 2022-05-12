#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
    transformer as fairseq_transformer,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LearnedPositionalEmbedding,
    MultiheadAttention,
    SinusoidalPositionalEmbedding,
)
from pytorch_translate import (
    transformer as pytorch_translate_transformer,
    vocab_reduction,
)
from pytorch_translate.average_attention import AverageAttention, AverageWindowAttention
from pytorch_translate.utils import torch_find


@register_model("transformer_aan")
class TransformerAANModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.
    """

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
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--encoder-learned-pos",
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
            action="store_true",
            help="use learned positional embeddings in the decoder",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--share-all-embeddings",
            action="store_true",
            help="share encoder, decoder and output embeddings"
            " (requires shared dictionary and embed dim)",
        )
        parser.add_argument(
            "--adaptive-softmax-cutoff",
            metavar="EXPR",
            help="comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion",
        ),
        parser.add_argument(
            "--adaptive-softmax-dropout",
            type=float,
            metavar="D",
            help="sets adaptive softmax dropout for the tail projections",
        )
        # AAN only
        parser.add_argument(
            "--decoder-attn-window-size",
            default=0,
            type=int,
            help="attention window size of the decoder (default: 0 (unlimited))",
        )
        parser.add_argument(
            "--no-decoder-aan-ffn",
            default=False,
            action="store_true",
            help="no FFN in the AAN block",
        )
        parser.add_argument(
            "--no-decoder-aan-gating",
            default=False,
            action="store_true",
            help="no Gating in the AAN block",
        )
        parser.add_argument(
            "--decoder-aan-ffn-use-embed-dim",
            default=False,
            action="store_true",
            help="""using decoder_embed_dim instead of decoder_ffn_embed_dim \
            as the hidden size of the FFN in AAN""",
        )
        parser.add_argument(
            "--decoder-aan-more-dropouts",
            type=lambda x: set(x.split(",")),
            help="""places to add more dropout in AAN, accepting multiple values in \
            [residual/after_avg/after_aan] separated by commas""",
        )
        parser.add_argument(
            "--decoder-out-embed-dim",
            default=None,
            type=int,
            metavar="N",
            help="decoder output embedding dimension (bottleneck layer before"
            "output layer if specified.)",
        )

        # Args for vocab reduction
        vocab_reduction.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 1024
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
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
                    """--share-all-embeddings requires --encoder-embed-dim \
                    to match --decoder-embed-dim"""
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise RuntimeError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = pytorch_translate_transformer.TransformerEncoder(
            args, src_dict, encoder_embed_tokens
        )
        decoder = TransformerAANDecoder(args, src_dict, tgt_dict, decoder_embed_tokens)
        return TransformerAANModel(task, encoder, decoder)

    def get_targets(self, sample, net_output):
        targets = sample["target"].view(-1)
        possible_translation_tokens = net_output[-1]
        if possible_translation_tokens is not None:
            targets = torch_find(
                possible_translation_tokens, targets, len(self.task.target_dictionary)
            )
        return targets


class TransformerAANDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerAANDecoderLayer`.
    """

    def __init__(
        self,
        args,
        src_dict,
        dst_dict,
        embed_tokens,
        no_encoder_attn=False,
        left_pad=False,
        final_norm=True,
    ):
        super().__init__(dst_dict)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = fairseq_transformer.PositionalEmbedding(
            1024, embed_dim, padding_idx, learned=args.decoder_learned_pos
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerAANDecoderLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )

        self.adaptive_softmax = None

        self.bottleneck_layer = None
        out_embed_dim = embed_dim
        if args.decoder_out_embed_dim is not None:
            assert (
                not args.share_all_embeddings
                and not args.share_decoder_input_output_embed
            ), "--decoder-out-embed-dim is incompatible with sharing output embeddings!"
            self.bottleneck_layer = Linear(embed_dim, args.decoder_out_embed_dim)
            out_embed_dim = args.decoder_out_embed_dim

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dst_dict),
                out_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dst_dict), out_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=out_embed_dim**-0.5)
        self.register_buffer("version", torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

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
        encoder_out=None,
        incremental_state=None,
        possible_translation_tokens=None,
        timestep=None,
    ):
        (encoder_x, src_tokens, encoder_padding_mask) = encoder_out

        positions = self.embed_positions(
            prev_output_tokens, incremental_state=incremental_state, timestep=timestep
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]

            if self.onnx_trace:
                assert type(incremental_state) is list
                assert timestep is not None

                state_list = incremental_state
                incremental_state = {}
                state_index = 0

                for layer in self.layers:
                    prev_sum = state_list[state_index]
                    state_index += 1
                    prev_pos = timestep.float()
                    layer.avg_attn._set_input_buffer(
                        incremental_state, {"prev_sum": prev_sum, "prev_pos": prev_pos}
                    )

                    if layer.encoder_attn is not None:
                        utils.set_incremental_state(
                            layer.encoder_attn,
                            incremental_state,
                            "prev_key",
                            state_list[state_index],
                        )
                        utils.set_incremental_state(
                            layer.encoder_attn,
                            incremental_state,
                            "prev_value",
                            state_list[state_index + 1],
                        )
                        state_index += 2

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_x,
                encoder_padding_mask,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x)
                if incremental_state is None
                else None,
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.bottleneck_layer is not None:
            x = self.bottleneck_layer(x)

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

        if self.adaptive_softmax is None:
            logits = F.linear(x, output_weights)
        else:
            assert (
                possible_translation_tokens is None
            ), "vocabulary reduction and adaptive softmax are incompatible!"
            logits = x

        if self.onnx_trace:
            state_outputs = []
            for layer in self.layers:
                saved_state = layer.avg_attn._get_input_buffer(incremental_state)
                assert (
                    "prev_sum" in saved_state
                ), "No prev_sum found while tracing average attention layer!"
                # remove sequence axis
                prev_sum = saved_state["prev_sum"].squeeze(0)
                state_outputs.append(prev_sum)

                if layer.encoder_attn is not None:
                    # prev_key and prev_value remain unchanged
                    prev_key = state_list[len(state_outputs)]
                    prev_value = state_list[len(state_outputs) + 1]
                    state_outputs.extend([prev_key, prev_value])

            return logits, attn, possible_translation_tokens, state_outputs

        return logits, attn, possible_translation_tokens

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

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

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if "decoder.embed_positions.weights" in state_dict:
                del state_dict["decoder.embed_positions.weights"]
            state_dict["decoder.embed_positions._float_tensor"] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                "0": "aan_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "decoder.layers.{}.layer_norms.{}.{}".format(i, old, m)
                    if k in state_dict:
                        state_dict[
                            "decoder.layers.{}.{}.{}".format(i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get("decoder.version", torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict["decoder.version"] = torch.Tensor([1])

        return state_dict

    def _init_prev_states(self, encoder_out):
        """
        For average attention, prev_sum is initialized to zero.
        For encoder-decoder attention, key and value are computed once from
        the encoder outputs and stay the same throughout decoding.
        """
        encoder_x, src_tokens, encoder_padding_mask = encoder_out
        states = []
        for layer in self.layers:
            # (bsz, channel)
            prev_sum = torch.zeros([1, layer.avg_attn.embed_dim])
            states.append(prev_sum)

            # (key, value) for encoder-decoder attention computed from encoder
            # output and remain the same throughout decoding
            key = layer.encoder_attn.k_proj(encoder_x)
            value = layer.encoder_attn.v_proj(encoder_x)

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


class TransformerAANDecoderLayer(nn.Module):
    """Decoder layer block.
    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.more_dropouts = args.decoder_aan_more_dropouts
        if args.decoder_attn_window_size <= 0:
            self.avg_attn = AverageAttention(
                self.embed_dim, dropout=args.attention_dropout
            )
        else:
            self.avg_attn = AverageWindowAttention(
                self.embed_dim,
                dropout=args.attention_dropout,
                window_size=args.decoder_attn_window_size,
            )
        # self.activation = getattr(args, "decoder_ffn_activation", "relu")
        self.aan_layer_norm = LayerNorm(self.embed_dim)
        if args.no_decoder_aan_ffn:
            self.aan_ffn = None
        else:
            aan_ffn_hidden_dim = (
                args.decoder_ffn_embed_dim
                if args.decoder_aan_ffn_use_embed_dim
                else args.decoder_ffn_embed_dim
            )
            self.aan_ffn = FeedForwardNetwork(
                self.embed_dim,
                aan_ffn_hidden_dim,
                self.embed_dim,
                num_layers=2,
                dropout=args.relu_dropout,
            )

        if args.no_decoder_aan_gating:
            self.aan_gating_fc = None
        else:
            self.aan_gating_fc = Linear(self.embed_dim * 2, self.embed_dim * 2)
        self.normalize_before = args.decoder_normalize_before

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=args.encoder_embed_dim,
                vdim=args.encoder_embed_dim,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.ffn = FeedForwardNetwork(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            num_layers=2,
            dropout=args.relu_dropout,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out,
        encoder_padding_mask,
        incremental_state,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        if "residual" in self.more_dropouts:
            residual = F.dropout(residual, p=self.dropout, training=self.training)

        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.avg_attn._set_input_buffer(incremental_state, saved_state)
        x, _ = self.avg_attn(
            value=x,
            mask_future_timesteps=True,
            incremental_state=incremental_state,
            mask_trick=self.training,
        )
        if "after_avg" in self.more_dropouts:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.aan_layer_norm is not None:
            x = self.maybe_layer_norm(self.aan_layer_norm, x, before=True)

        if self.aan_ffn is not None:
            x = self.aan_ffn(x)
            if "after_ffn" in self.more_dropouts:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.aan_gating_fc is not None:
            i, f = self.aan_gating_fc(torch.cat([residual, x], dim=-1)).chunk(2, dim=-1)
            x = torch.sigmoid(f) * residual + torch.sigmoid(i) * x
            if "after_gating" in self.more_dropouts:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        if self.aan_layer_norm is not None:
            x = self.maybe_layer_norm(self.aan_layer_norm, x, after=True)

        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.ffn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def extra_repr(self):
        return "dropout={}, more_dropouts={}".format(self.dropout, self.more_dropouts)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def PositionalEmbeddingCreator(
    num_embeddings, embedding_dim, padding_idx, left_pad, learned=False
):
    if learned:
        m = LearnedPositionalEmbedding(
            num_embeddings + padding_idx + 1, embedding_dim, padding_idx
        )
        nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_size=num_embeddings + padding_idx + 1,
        )
    return m


class FeedForwardNetwork(nn.Module):
    def __init__(
        self, input_size, hidden_size=None, output_size=None, num_layers=2, dropout=0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = self.input_size if output_size is None else output_size
        self.hidden_size = self.output_size if hidden_size is None else hidden_size
        self.num_layers = num_layers
        self.activation_type = "relu"
        self.dropout = dropout
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(Linear(self.input_size, self.output_size))
        else:
            self.layers.append(Linear(self.input_size, self.hidden_size))
            for _ in range(1, num_layers - 1):
                self.layers.append(Linear(self.hidden_size, self.hidden_size))
            self.layers.append(Linear(self.hidden_size, self.output_size))

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x)
        return x

    def extra_repr(self):
        return "activation_type={}, dropout={}".format(
            self.activation_type, self.dropout
        )


@register_model_architecture("transformer_aan", "transformer_aan")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", None)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.decoder_aan_ffn = getattr(args, "decoder_aan_ffn", True)
    args.decoder_aan_ffn_use_embed_dim = getattr(
        args, "decoder_aan_ffn_use_embed_dim", False
    )
    args.decoder_aan_gating = getattr(args, "decoder_aan_gating", True)
    args.decoder_aan_more_dropouts = getattr(args, "decoder_aan_more_dropouts", "")
    vocab_reduction.set_arg_defaults(args)
