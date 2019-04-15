#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqModel,
    register_model,
    register_model_architecture,
    transformer as fairseq_transformer,
)
from pytorch_translate import (
    transformer as pytorch_translate_transformer,
    vocab_reduction,
)
from pytorch_translate.common_layers import TransformerTokenEmbedding
from pytorch_translate.utils import torch_find


@register_model("hybrid_transformer_rnn")
class HybridTransformerRNNModel(FairseqModel):
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
            "--decoder-reduced-attention-dim",
            type=int,
            default=None,
            metavar="N",
            help="if specified, computes attention with this dimensionality "
            "(instead of using encoder output dims)",
        )
        parser.add_argument(
            "--decoder-lstm-units",
            type=int,
            metavar="N",
            help="num LSTM units for each decoder layer",
        )
        parser.add_argument(
            "--decoder-out-embed-dim",
            type=int,
            metavar="N",
            help="decoder output embedding dimension",
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

        encoder_embed_tokens = pytorch_translate_transformer.build_embedding(
            dictionary=src_dict,
            embed_dim=args.encoder_embed_dim,
            path=args.encoder_pretrained_embed,
            freeze=args.encoder_freeze_embed,
        )
        decoder_embed_tokens = pytorch_translate_transformer.build_embedding(
            dictionary=tgt_dict,
            embed_dim=args.decoder_embed_dim,
            path=args.decoder_pretrained_embed,
            freeze=args.decoder_freeze_embed,
        )

        encoder = pytorch_translate_transformer.TransformerEncoder(
            args, src_dict, encoder_embed_tokens, proj_to_decoder=False
        )
        decoder = HybridRNNDecoder(args, src_dict, tgt_dict, decoder_embed_tokens)
        return HybridTransformerRNNModel(task, encoder, decoder)

    def get_targets(self, sample, net_output):
        targets = sample["target"].view(-1)
        possible_translation_tokens = net_output[-1]
        if possible_translation_tokens is not None:
            targets = torch_find(
                possible_translation_tokens, targets, len(self.task.target_dictionary)
            )
        return targets


class HybridRNNDecoder(FairseqIncrementalDecoder):
    """
    Decoder with general structure of Chen et al., The Best of Both Worlds:
    Combining Recent Advances in Neural Machine Translation, 2018.
    https://arxiv.org/abs/1804.09849
    """

    def _init_dims(self, args, src_dict, dst_dict, embed_tokens):
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.embed_tokens = embed_tokens

        self.lstm_units = args.decoder_lstm_units
        self.num_layers = args.decoder_layers
        self.initial_input_dim = embed_dim

        self.encoder_output_dim = args.encoder_embed_dim
        if args.decoder_reduced_attention_dim is None:
            self.attention_dim = self.encoder_output_dim
        else:
            self.attention_dim = args.decoder_reduced_attention_dim
        self.input_dim = self.lstm_units + self.attention_dim

        self.num_attention_heads = args.decoder_attention_heads
        self.out_embed_dim = args.decoder_out_embed_dim

    def _init_components(self, args, src_dict, dst_dict, embed_tokens):
        self.initial_rnn_layer = nn.LSTM(
            input_size=self.initial_input_dim, hidden_size=self.lstm_units
        )

        self.proj_encoder_layer = None
        if self.attention_dim != self.encoder_output_dim:
            self.proj_encoder_layer = fairseq_transformer.Linear(
                self.encoder_output_dim, self.attention_dim
            )

        self.proj_layer = None
        if self.lstm_units != self.attention_dim:
            self.proj_layer = fairseq_transformer.Linear(
                self.lstm_units, self.attention_dim
            )

        self.attention = fairseq_transformer.MultiheadAttention(
            self.attention_dim, self.num_attention_heads, dropout=args.attention_dropout
        )

        self.extra_rnn_layers = nn.ModuleList([])
        for _ in range(self.num_layers - 1):
            self.extra_rnn_layers.append(
                nn.LSTM(input_size=self.input_dim, hidden_size=self.lstm_units)
            )

        self.bottleneck_layer = fairseq_transformer.Linear(
            self.input_dim, self.out_embed_dim
        )

        self.embed_out = nn.Parameter(torch.Tensor(len(dst_dict), self.out_embed_dim))
        nn.init.normal_(self.embed_out, mean=0, std=self.out_embed_dim ** -0.5)

        self.vocab_reduction_module = None
        if args.vocab_reduction_params:
            self.vocab_reduction_module = vocab_reduction.VocabReduction(
                src_dict, dst_dict, args.vocab_reduction_params, fp16=args.fp16
            )

        self.onnx_trace = False

    def __init__(self, args, src_dict, dst_dict, embed_tokens):
        super().__init__(dst_dict)
        self._init_dims(args, src_dict, dst_dict, embed_tokens)
        self._init_components(args, src_dict, dst_dict, embed_tokens)

    # Enable dependency injection by subclasses
    def _unpack_encoder_out(self, encoder_out):
        """ Allow taking encoder_out from different architecture which
        may have different formats.
        """
        return encoder_out

    def _init_hidden(self, encoder_out, batch_size):
        """ Initialize with latent code if available otherwise zeros."""
        return torch.zeros([1, batch_size, self.lstm_units])

    def _concat_latent_code(self, x, encoder_out):
        """ Concat latent code, if available in encoder_out, which is the
        case in subclass.
        """
        return x

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
        (encoder_x, src_tokens, encoder_padding_mask) = self._unpack_encoder_out(
            encoder_out
        )

        bsz, seqlen = prev_output_tokens.size()
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        state_outputs = []
        if incremental_state is not None:
            prev_states = utils.get_incremental_state(
                self, incremental_state, "cached_state"
            )
            if prev_states is None:
                prev_states = self._init_prev_states(encoder_out)

            # final 2 states of list are projected key and value
            saved_state = {"prev_key": prev_states[-2], "prev_value": prev_states[-1]}
            self.attention._set_input_buffer(incremental_state, saved_state)

        if incremental_state is not None:
            # first num_layers pairs of states are (prev_hidden, prev_cell)
            # for each layer
            h_prev = prev_states[0]
            c_prev = prev_states[1]
        else:
            h_prev = self._init_hidden(encoder_out, bsz).type_as(x)
            c_prev = torch.zeros([1, bsz, self.lstm_units]).type_as(x)

        x = self._concat_latent_code(x, encoder_out)
        x, (h_next, c_next) = self.initial_rnn_layer(x, (h_prev, c_prev))
        if incremental_state is not None:
            state_outputs.extend([h_next, c_next])

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.proj_encoder_layer is not None:
            encoder_x = self.proj_encoder_layer(encoder_x)

        attention_in = x
        if self.proj_layer is not None:
            attention_in = self.proj_layer(x)

        attention_out, attention_weights = self.attention(
            query=attention_in,
            key=encoder_x,
            value=encoder_x,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=(not self.training),
        )

        for i, layer in enumerate(self.extra_rnn_layers):
            residual = x
            rnn_input = torch.cat([x, attention_out], dim=2)
            rnn_input = self._concat_latent_code(rnn_input, encoder_out)

            if incremental_state is not None:
                # first num_layers pairs of states are (prev_hidden, prev_cell)
                # for each layer
                h_prev = prev_states[2 * i + 2]
                c_prev = prev_states[2 * i + 3]
            else:
                h_prev = self._init_hidden(encoder_out, bsz).type_as(x)
                c_prev = torch.zeros([1, bsz, self.lstm_units]).type_as(x)

            x, (h_next, c_next) = layer(rnn_input, (h_prev, c_prev))
            if incremental_state is not None:
                state_outputs.extend([h_next, c_next])
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        x = torch.cat([x, attention_out], dim=2)
        x = self._concat_latent_code(x, encoder_out)
        x = self.bottleneck_layer(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if (
            self.vocab_reduction_module is not None
            and possible_translation_tokens is None
        ):
            decoder_input_tokens = prev_output_tokens.contiguous()
            possible_translation_tokens = self.vocab_reduction_module(
                src_tokens, decoder_input_tokens=decoder_input_tokens
            )

        output_weights = self.embed_out
        if possible_translation_tokens is not None:
            output_weights = output_weights.index_select(
                dim=0, index=possible_translation_tokens
            )

        logits = F.linear(x, output_weights)

        if incremental_state is not None:
            # encoder projections can be reused at each incremental step
            state_outputs.extend([prev_states[-2], prev_states[-1]])
            utils.set_incremental_state(
                self, incremental_state, "cached_state", state_outputs
            )

        return logits, attention_weights, possible_translation_tokens

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def _init_prev_states(self, encoder_out):
        """
        Initial (hidden, cell) values for LSTM layers are zero.

        For encoder-decoder attention, key and value are computed once from
        the encoder outputs and stay the same throughout decoding.
        """
        (encoder_x, src_tokens, encoder_padding_mask) = self._unpack_encoder_out(
            encoder_out
        )
        batch_size = torch.onnx.operators.shape_as_tensor(encoder_x)[1]

        if self.proj_encoder_layer is not None:
            encoder_x = self.proj_encoder_layer(encoder_x)

        states = []
        for _ in range(self.num_layers):
            hidden = self._init_hidden(encoder_out, batch_size).type_as(encoder_x)
            cell = torch.zeros([1, batch_size, self.lstm_units]).type_as(encoder_x)
            states.extend([hidden, cell])

        # (key, value) for encoder-decoder attention computed from encoder
        # output and remain the same throughout decoding
        key = self.attention.in_proj_k(encoder_x)
        value = self.attention.in_proj_v(encoder_x)

        # (key, value) kept in shape (bsz, num_heads, seq_len, head_dim)
        # to avoid repeated transpose operations
        seq_len, batch_size_int, _ = encoder_x.shape
        num_heads = self.attention.num_heads
        head_dim = self.attention.head_dim
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

    def reorder_incremental_state(self, incremental_state, new_order):
        # parent reorders attention model
        super().reorder_incremental_state(incremental_state, new_order)

        cached_state = utils.get_incremental_state(
            self, incremental_state, "cached_state"
        )
        if cached_state is None:
            return

        # Last 2 elements of prev_states are encoder projections
        # used for ONNX export
        for i, state in enumerate(cached_state[:-2]):
            cached_state[i] = state.index_select(1, new_order)

        utils.set_incremental_state(
            self, incremental_state, "cached_state", cached_state
        )


@register_model_architecture("hybrid_transformer_rnn", "hybrid_transformer_rnn")
def base_architecture(args):
    args.encoder_pretrained_embed = getattr(args, "encoder_pretrained_embed", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_freeze_embed = getattr(args, "encoder_freeze_embed", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_pretrained_embed = getattr(args, "decoder_pretrained_embed", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_reduced_attention_dim = getattr(
        args, "decoder_reduced_attention_dim", None
    )
    args.decoder_lstm_units = getattr(args, "decoder_lstm_units", 512)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 256)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    vocab_reduction.set_arg_defaults(args)
