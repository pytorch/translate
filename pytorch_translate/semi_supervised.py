#!/usr/bin/env python3

import copy
from collections import OrderedDict

import torch.nn as nn
from fairseq.models import (
    FairseqMultiModel,
    register_model,
    register_model_architecture,
)
from pytorch_translate import constants
from pytorch_translate.rnn import RNNModel, base_architecture
from pytorch_translate.tasks.semi_supervised_task import PytorchTranslateSemiSupervised


@register_model("semi_supervised")
class SemiSupervisedModel(FairseqMultiModel):
    """Train RNN models with iterative backtranslations.
    """

    def __init__(self, task, encoders, decoders):
        super().__init__(encoders, decoders)
        self.task = task

        # TODO(T35638969): pass this class into super (FairseqMultiModel)
        self.models = nn.ModuleDict(
            # Need to use RNNModel since FairseqModel doesn't define get_targets
            {key: RNNModel(task, encoders[key], decoders[key]) for key in self.keys}
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # TODO(T35638969): Generalize this to be able to use other model classes
        # like Transformer
        RNNModel.add_args(parser)
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

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        assert isinstance(task, PytorchTranslateSemiSupervised)

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

        def strip_suffix(lang):
            """
            Both "lang" and "lang_mono" languages share the same encoder/decoder
            since they belong to the same language but use bilingual and monolingual
            corpora respectively to train
            So use "lang" as model key for both "lang" and "lang_mono" by stripping
            the suffix "_mono" if it exists
            """
            if f"_{constants.MONOLINGUAL_DATA_IDENTIFIER}" in lang:
                lang = lang[: -(len(f"_{constants.MONOLINGUAL_DATA_IDENTIFIER}"))]
            return lang

        """
        TODO(T35638969): Generalize this to be able to use other model classes
        like Transformer TransformerModel does not currently have build_encoder
        and build_decoder methods
        """

        def get_encoder(lang):
            lang = strip_suffix(lang)
            if lang not in lang_encoders:
                lang_encoders[lang] = RNNModel.build_encoder(args, task.dicts[lang])
            return lang_encoders[lang]

        def get_decoder(lang_pair):
            """
            Fetch decoder for the input `lang_pair`, which denotes the target
            language of the model
            """
            source_lang, target_lang = (
                strip_suffix(lang) for lang in lang_pair.split("-")
            )
            if target_lang not in lang_decoders:
                # hack to prevent VR for denoising autoencoder. We remove vocab
                # reduction params if we have lang-lang_any_suffix
                args_maybe_modified = copy.deepcopy(args)
                if source_lang == target_lang:
                    args_maybe_modified.vocab_reduction_params = None

                lang_decoders[target_lang] = RNNModel.build_decoder(
                    args_maybe_modified,
                    task.dicts[source_lang],
                    task.dicts[target_lang],
                )
            return lang_decoders[target_lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src in zip(task.lang_pairs, src_langs):
            encoders[lang_pair] = (
                shared_encoder if shared_encoder is not None else get_encoder(src)
            )
            decoders[lang_pair] = (
                shared_decoder if shared_decoder is not None else get_decoder(lang_pair)
            )

        return SemiSupervisedModel(task, encoders, decoders)


@register_model_architecture("semi_supervised", "semi_supervised")
def semi_supervised(args):
    base_architecture(args)
    args.share_encoder_embeddings = getattr(args, "share_encoder_embeddings", False)
    args.share_decoder_embeddings = getattr(args, "share_decoder_embeddings", False)
    args.share_encoders = getattr(args, "share_encoders", False)
    args.share_decoders = getattr(args, "share_decoders", False)
