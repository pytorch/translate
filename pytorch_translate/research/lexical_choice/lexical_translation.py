#!/usr/bin/env python3

import torch


def attention_weighted_src_embedding(
    src_embedding, attn_scores, activation_fn=torch.tanh
):
    """
    use the attention weights to form a weighted average of embeddings
    parameters:
        src_embedding:  srclen x bsz x embeddim
        attn_scores: bsz x tgtlen x srclen
    return:
        lex: bsz x tgtlen x embeddim
    """
    # lexical choice varying lengths: T x B x C -> B x T x C
    src_embedding = src_embedding.transpose(0, 1)

    lex = torch.bmm(attn_scores, src_embedding)
    lex = activation_fn(lex)
    return lex


def lex_logits(lex_h, output_projection_w_lex, output_projection_b_lex, logits_shape):
    """
    calculate the logits of lexical layer output
    """
    projection_lex_flat = torch.matmul(output_projection_w_lex, lex_h.t()).t()

    logits = (
        torch.onnx.operators.reshape_from_tensor_shape(
            projection_lex_flat, logits_shape
        )
        + output_projection_b_lex
    )
    return logits
