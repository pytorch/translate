#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from pytorch_translate import attention, utils as pytorch_translate_utils
from pytorch_translate.common_layers import (
    DecoderWithOutputProjection,
    Embedding,
    Linear,
    NonlinearLayer,
)
from pytorch_translate.utils import maybe_cat


class NGramDecoder(DecoderWithOutputProjection):
    """n-gram decoder.

    This decoder implementation does not condition on the full target-side
    history. Instead, predictions only depend on the target n-gram history and
    the full source sentence via attention over encoder outputs. The decoder
    network is a feedforward network with source context as additional input.
    """

    def __init__(
        self,
        src_dict,
        dst_dict,
        vocab_reduction_params=None,
        n=4,
        encoder_hidden_dim=512,
        embed_dim=512,
        freeze_embed=False,
        hidden_dim=512,
        out_embed_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        attention_type="dot",
        residual_level=None,
        activation_fn=nn.ReLU,
        project_output=True,
        pretrained_embed=None,
        projection_pretrained_embed=None,
    ):
        super().__init__(
            src_dict,
            dst_dict,
            vocab_reduction_params,
            out_embed_dim,
            project_output=project_output,
            pretrained_embed=projection_pretrained_embed,
        )
        self.history_len = n - 1
        self.encoder_hidden_dim = encoder_hidden_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_embed_dim = out_embed_dim
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.attention_type = attention_type
        self.residual_level = residual_level
        self.dst_dict = dst_dict
        self.activation_fn = activation_fn

        num_embeddings = len(dst_dict)
        padding_idx = dst_dict.pad()
        self.embed_tokens = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
            freeze_embed=freeze_embed,
        )
        pytorch_translate_utils.load_embedding(
            embedding=self.embed_tokens,
            dictionary=dst_dict,
            pretrained_embed=pretrained_embed,
        )

        self.history_conv = nn.Sequential(
            torch.nn.Conv1d(embed_dim, hidden_dim, self.history_len), activation_fn()
        )

        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            [
                NonlinearLayer(hidden_dim, hidden_dim, activation_fn=activation_fn)
                for _ in range(num_layers)
            ]
        )

        self.attention = attention.build_attention(
            attention_type=attention_type,
            decoder_hidden_state_dim=hidden_dim,
            context_dim=encoder_hidden_dim,
            force_projection=True,
        )
        self.combined_output_and_context_dim = self.attention.context_dim + hidden_dim
        if self.combined_output_and_context_dim != out_embed_dim:
            self.additional_fc = Linear(
                self.combined_output_and_context_dim, out_embed_dim
            )

    def forward_unprojected(self, input_tokens, encoder_out, incremental_state=None):
        padded_tokens = F.pad(
            input_tokens,
            (self.history_len - 1, 0, 0, 0),
            "constant",
            self.dst_dict.eos(),
        )
        # We use incremental_state only to check whether we are decoding or not
        # self.training is false even for the forward pass through validation
        if incremental_state is not None:
            padded_tokens = padded_tokens[:, -self.history_len :]
        utils.set_incremental_state(self, incremental_state, "incremental_marker", True)

        bsz, seqlen = padded_tokens.size()
        seqlen -= self.history_len - 1

        # get outputs from encoder
        (encoder_outs, final_hidden, _, src_lengths, _) = encoder_out

        # padded_tokens has shape [batch_size, seq_len+history_len]
        x = self.embed_tokens(padded_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # Convolution needs shape [batch_size, channels, seq_len]
        x = self.history_conv(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, p=self.dropout_out, training=self.training)

        # x has shape [batch_size, seq_len, channels]
        for i, layer in enumerate(self.layers):
            prev_x = x
            x = layer(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
            if self.residual_level is not None and i >= self.residual_level:
                x = x + prev_x

        # Attention
        attn_out, attn_scores = self.attention(
            x.transpose(0, 1).contiguous().view(-1, self.hidden_dim),
            encoder_outs.repeat(1, seqlen, 1),
            src_lengths.repeat(seqlen),
        )
        if attn_out is not None:
            attn_out = attn_out.view(seqlen, bsz, -1).transpose(1, 0)
        attn_scores = attn_scores.view(-1, seqlen, bsz).transpose(0, 2)
        x = maybe_cat((x, attn_out), dim=2)

        # bottleneck layer
        if hasattr(self, "additional_fc"):
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
        return x, attn_scores

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number
