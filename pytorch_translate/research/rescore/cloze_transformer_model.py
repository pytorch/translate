#!/usr/bin/env python3

import torch
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from pytorch_translate.transformer import (
    base_architecture,
    build_embedding,
    TransformerDecoder,
    TransformerModel,
)


@register_model("cloze_transformer")
class ClozeTransformerModel(TransformerModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        cloze_transformer_architecture(args)

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

        encoder = ClozeTransformerModel.build_encoder(
            args, src_dict, embed_tokens=encoder_embed_tokens
        )
        decoder = ClozeTransformerModel.build_decoder(
            args, src_dict, tgt_dict, embed_tokens=decoder_embed_tokens
        )
        return ClozeTransformerModel(task, encoder, decoder)

    @classmethod
    def build_decoder(cls, args, src_dict, dst_dict, embed_tokens):
        return ClozeTransformerDecoder(
            args, src_dict, dst_dict, embed_tokens=embed_tokens
        )


class ClozeTransformerDecoder(TransformerDecoder):
    """Cloze-Transformer decoder."""

    def __init__(self, args, src_dict, dst_dict, embed_tokens, left_pad=False):
        super().__init__(args, src_dict, dst_dict, embed_tokens)
        assert args.decoder_layers == 1

    def buffered_future_mask(self, tensor):
        """attend all surounding words except itself
           [[0, -inf, 0]
            [0,  0, -inf]
            [0,  0,   0]]
        The attention map is not ture diagonal since we predict y_{t+1} at time-step t
        """
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
            self._future_mask = torch.tril(self._future_mask, 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
            )
            self._future_mask = torch.tril(self._future_mask, 1)
        return self._future_mask[:dim, :dim]


@register_model_architecture("cloze_transformer", "cloze_transformer")
def cloze_transformer_architecture(args):
    base_architecture(args)
