#!/usr/bin/env python3

from collections import OrderedDict

import torch.nn as nn
from fairseq.models import FairseqMultiModel, register_model
from pytorch_translate import common_layers, utils


@register_model("multilingual")
class MultilingualModel(FairseqMultiModel):
    """
    To use, you must extend this class and define single_model_cls as a class
    variable. Example:

        @register_model("multilingual_transformer")
        class MultilingualTransformerModel(MultilingualModel):
            single_model_cls = TransformerModel

            @staticmethod
            def add_args(parser):
                TransformerModel.add_args(parser)
                MultilingualModel.add_args(parser)
    """

    def __init__(self, task, encoders, decoders):
        super().__init__(encoders, decoders)
        self.task = task
        self.models = nn.ModuleDict(
            {
                key: self.__class__.single_model_cls(task, encoders[key], decoders[key])
                for key in self.keys
            }
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--share-encoder-embeddings",
            action="store_true",
            help="share encoder embeddings across languages",
        )
        parser.add_argument(
            "--share-decoder-embeddings",
            action="store_true",
            help="share decoder embeddings across languages",
        )
        parser.add_argument(
            "--share-encoders",
            action="store_true",
            help="share encoders across languages",
        )
        parser.add_argument(
            "--share-decoders",
            action="store_true",
            help="share decoders across languages",
        )

    @staticmethod
    def set_multilingual_arch_args(args):
        args.share_encoder_embeddings = getattr(args, "share_encoder_embeddings", False)
        args.share_decoder_embeddings = getattr(args, "share_decoder_embeddings", False)
        args.share_encoders = getattr(args, "share_encoders", False)
        args.share_decoders = getattr(args, "share_decoders", False)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 1024
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 1024

        src_langs = [lang_pair.split("-")[0] for lang_pair in task.lang_pairs]
        tgt_langs = [lang_pair.split("-")[1] for lang_pair in task.lang_pairs]

        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(lang, shared_encoder_embed_tokens=None):
            if lang not in lang_encoders:
                src_dict = task.dicts[lang]
                if shared_encoder_embed_tokens is None:
                    encoder_embed_tokens = common_layers.Embedding(
                        num_embeddings=len(src_dict),
                        embedding_dim=args.encoder_embed_dim,
                        padding_idx=src_dict.pad(),
                        freeze_embed=args.encoder_freeze_embed,
                        normalize_embed=getattr(args, "encoder_normalize_embed", False),
                    )
                    utils.load_embedding(
                        embedding=encoder_embed_tokens,
                        dictionary=src_dict,
                        pretrained_embed=args.encoder_pretrained_embed,
                    )
                else:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                lang_encoders[lang] = cls.single_model_cls.build_encoder(
                    args, src_dict, embed_tokens=encoder_embed_tokens
                )
            return lang_encoders[lang]

        def get_decoder(lang, shared_decoder_embed_tokens=None):
            """
            Fetch decoder for the input `lang`, which denotes the target
            language of the model
            """
            if lang not in lang_decoders:
                tgt_dict = task.dicts[lang]
                if shared_decoder_embed_tokens is None:
                    decoder_embed_tokens = common_layers.Embedding(
                        num_embeddings=len(tgt_dict),
                        embedding_dim=args.decoder_embed_dim,
                        padding_idx=tgt_dict.pad(),
                        freeze_embed=args.decoder_freeze_embed,
                    )
                    utils.load_embedding(
                        embedding=decoder_embed_tokens,
                        dictionary=tgt_dict,
                        pretrained_embed=args.decoder_pretrained_embed,
                    )
                else:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                lang_decoders[lang] = cls.single_model_cls.build_decoder(
                    args, task.dicts[lang], tgt_dict, embed_tokens=decoder_embed_tokens
                )
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_encoder_embeddings:
            shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=src_langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=common_layers.build_embedding,
                pretrained_embed_path=None,
            )
        if args.share_decoder_embeddings:
            shared_decoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=tgt_langs,
                embed_dim=args.decoder_embed_dim,
                build_embedding=common_layers.build_embedding,
                pretrained_embed_path=None,
            )
        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src_lang, tgt_lang in zip(task.lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = (
                shared_encoder
                if shared_encoder is not None
                else get_encoder(
                    src_lang, shared_encoder_embed_tokens=shared_encoder_embed_tokens
                )
            )
            decoders[lang_pair] = (
                shared_decoder
                if shared_decoder is not None
                else get_decoder(
                    tgt_lang, shared_decoder_embed_tokens=shared_decoder_embed_tokens
                )
            )

        return cls(task, encoders, decoders)
