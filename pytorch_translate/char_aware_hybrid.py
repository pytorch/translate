#!/usr/bin/env python3

from ast import literal_eval

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_translate import char_encoder, hybrid_transformer_rnn


class CharAwareHybridRNNDecoder(hybrid_transformer_rnn.HybridRNNDecoder):
    """
        A decoder that is similar to the HybridRNNDecoder but has a character
        CNN encoder to get the representation for each generated previous token.
        The decoder is similar to https://arxiv.org/pdf/1809.02223.pdf.
    """

    def __init__(
        self,
        args,
        src_dict,
        dst_dict,
        embed_tokens,
        num_chars=50,
        char_embed_dim=32,
        char_cnn_params="[(128, 3), (128, 5)]",
        char_cnn_nonlinear_fn="tanh",
        char_cnn_num_highway_layers=0,
        use_pretrained_weights=False,
        finetune_pretrained_weights=False,
    ):
        super().__init__(args, src_dict, dst_dict, embed_tokens)
        convolutions_params = literal_eval(char_cnn_params)
        self.char_cnn_encoder = char_encoder.CharCNNModel(
            dictionary=dst_dict,
            num_chars=num_chars,
            char_embed_dim=char_embed_dim,
            convolutions_params=convolutions_params,
            nonlinear_fn_type=char_cnn_nonlinear_fn,
            num_highway_layers=char_cnn_num_highway_layers,
            # char_cnn_output_dim should match the word embedding dimension.
            char_cnn_output_dim=embed_tokens.embedding_dim,
            use_pretrained_weights=use_pretrained_weights,
            finetune_pretrained_weights=finetune_pretrained_weights,
        )
        self.char_layer_norm = nn.LayerNorm(embed_tokens.embedding_dim)

    def _embed_prev_outputs(
        self, prev_output_tokens, incremental_state=None, prev_output_chars=None
    ):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if prev_output_chars is not None:
                prev_output_chars = prev_output_chars[:, -1:, :].squeeze(1)

        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        bsz, maxchars = prev_output_chars.size()
        # char_cnn_encoder takes input (max_word_length, total_words)
        char_inds_flat = prev_output_chars.view(-1, maxchars).t()
        # output (total_words, encoder_dim)
        char_cnn_output = self.char_cnn_encoder(char_inds_flat)

        char_cnn_output = char_cnn_output.view(bsz, 1, char_cnn_output.shape[-1])
        # (seqlen, bsz, char_cnn_output_dim)
        char_cnn_output = char_cnn_output.transpose(0, 1)
        char_cnn_output = self.char_layer_norm(char_cnn_output)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        return torch.add(x, char_cnn_output), prev_output_tokens

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state=None,
        possible_translation_tokens=None,
        timestep=None,
        prev_output_chars=None,
    ):
        """
        The assumption is that during inference, the word embedding values are
        summed with their corresponding character representations. Thus the model
        will look like the same as a word-based decoder.
        """
        if self.training:
            x, prev_output_tokens = self._embed_prev_outputs(
                prev_output_tokens=prev_output_tokens,
                incremental_state=incremental_state,
                prev_output_chars=prev_output_chars,
            )
        else:
            x, prev_output_tokens = super()._embed_prev_outputs(
                prev_output_tokens=prev_output_tokens,
                incremental_state=incremental_state,
            )
        return self._forward_given_embeddings(
            embed_out=x,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            possible_translation_tokens=possible_translation_tokens,
            timestep=timestep,
        )
