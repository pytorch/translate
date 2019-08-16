#!/usr/bin/env python3

from ast import literal_eval

import torch.nn as nn
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
        char_cnn_pool_type="max",
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
            pool_type=char_cnn_pool_type,
            num_highway_layers=char_cnn_num_highway_layers,
            # char_cnn_output_dim should match the word embedding dimension.
            char_cnn_output_dim=embed_tokens.embedding_dim,
            use_pretrained_weights=use_pretrained_weights,
            finetune_pretrained_weights=finetune_pretrained_weights,
        )
        self.char_layer_norm = nn.LayerNorm(embed_tokens.embedding_dim)
