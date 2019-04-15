#!/usr/bin/env python3

import logging
import math
from ast import literal_eval

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
    transformer as fairseq_transformer,
)
from fairseq.modules import SinusoidalPositionalEmbedding
from pytorch_translate import char_encoder, transformer, utils, vocab_constants
from pytorch_translate.char_source_model import verify_pretrain_params
from pytorch_translate.common_layers import (
    TransformerEncoderGivenEmbeddings,
    VariableTracker,
)
from pytorch_translate.dictionary import TAGS


logger = logging.getLogger(__name__)


@register_model("char_source_transformer")
class CharSourceTransformerModel(transformer.TransformerModel):
    def __init__(self, task, encoder, decoder):
        super().__init__(task, encoder, decoder)

    @staticmethod
    def add_args(parser):
        transformer.TransformerModel.add_args(parser)
        parser.add_argument(
            "--char-embed-dim",
            type=int,
            default=128,
            metavar="N",
            help=("Character embedding dimension."),
        )
        parser.add_argument(
            "--char-cnn-params",
            type=str,
            metavar="EXPR",
            help=("String experission, [(dim, kernel_size), ...]."),
        )
        parser.add_argument(
            "--char-cnn-nonlinear-fn",
            type=str,
            default="tanh",
            metavar="EXPR",
            help=("Nonlinearity applied to char conv outputs. Values: relu, tanh."),
        )
        parser.add_argument(
            "--char-cnn-pool-type",
            type=str,
            default="max",
            metavar="EXPR",
            help=(
                "Pooling function of input sequence outputs. "
                "Values: logsumexp, max, mean, meanmax."
            ),
        )
        parser.add_argument(
            "--char-cnn-num-highway-layers",
            type=int,
            default=0,
            metavar="N",
            help=("Char cnn encoder highway layers."),
        )
        parser.add_argument(
            "--char-cnn-output-dim",
            type=int,
            default=-1,
            metavar="N",
            help="Output dim of the CNN layer. If set to -1, this is computed "
            "from char-cnn-params.",
        )
        parser.add_argument(
            "--use-pretrained-weights",
            type=utils.bool_flag,
            nargs="?",
            const=True,
            default=False,
            help="Use pretrained weights for the character model including "
            "the char embeddings, CNN filters, highway networks",
        )
        parser.add_argument(
            "--finetune-pretrained-weights",
            type=utils.bool_flag,
            nargs="?",
            const=True,
            default=False,
            help="Boolean flag to specify whether or not to update the "
            "pretrained weights as part of training",
        )
        parser.add_argument(
            "--pretrained-weights-file",
            type=str,
            default="",
            help=("Weights file for loading pretrained weights"),
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        src_dict, dst_dict = task.source_dictionary, task.target_dictionary
        base_architecture(args)

        assert hasattr(args, "char_source_dict_size"), (
            "args.char_source_dict_size required. "
            "should be set by load_binarized_dataset()"
        )

        if args.share_all_embeddings:
            if src_dict != dst_dict:
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
            encoder_embed_tokens = transformer.build_embedding(
                src_dict,
                args.encoder_embed_dim,
                args.encoder_pretrained_embed,
                args.encoder_freeze_embed,
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = transformer.build_embedding(
                src_dict,
                args.encoder_embed_dim,
                args.encoder_pretrained_embed,
                args.encoder_freeze_embed,
            )
            decoder_embed_tokens = transformer.build_embedding(
                dst_dict,
                args.decoder_embed_dim,
                args.decoder_pretrained_embed,
                args.decoder_freeze_embed,
            )

        args.embed_bytes = getattr(args, "embed_bytes", False)

        # If we embed bytes then the number of indices is fixed and does not
        # depend on the dictionary
        if args.embed_bytes:
            num_chars = vocab_constants.NUM_BYTE_INDICES + TAGS.__len__() + 1
        else:
            num_chars = args.char_source_dict_size

        # In case use_pretrained_weights is true, verify the model params
        # are correctly set
        if args.embed_bytes and getattr(args, "use_pretrained_weights", False):
            verify_pretrain_params(args)

        encoder = CharCNNEncoder(
            args,
            src_dict,
            encoder_embed_tokens,
            num_chars=num_chars,
            embed_dim=args.char_embed_dim,
            char_cnn_params=args.char_cnn_params,
            char_cnn_nonlinear_fn=args.char_cnn_nonlinear_fn,
            char_cnn_pool_type=args.char_cnn_pool_type,
            char_cnn_num_highway_layers=args.char_cnn_num_highway_layers,
            char_cnn_output_dim=getattr(args, "char_cnn_output_dim", -1),
            use_pretrained_weights=getattr(args, "use_pretrained_weights", False),
            finetune_pretrained_weights=getattr(
                args, "finetune_pretrained_weights", False
            ),
            weights_file=getattr(args, "pretrained_weights_file", ""),
        )
        decoder = transformer.TransformerDecoder(
            args=args,
            src_dict=src_dict,
            dst_dict=dst_dict,
            embed_tokens=decoder_embed_tokens,
        )
        return cls(task, encoder, decoder)

    def forward(
        self, src_tokens, src_lengths, char_inds, word_lengths, prev_output_tokens
    ):
        """
        Overriding FairseqModel.forward() due to different encoder inputs.
        """
        encoder_out = self.encoder(src_tokens, src_lengths, char_inds, word_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out


class CharCNNEncoder(FairseqEncoder):
    """
    Character-level CNN encoder to generate word representations, as input to
    transformer encoder.
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        num_chars=50,
        embed_dim=32,
        char_cnn_params="[(128, 3), (128, 5)]",
        char_cnn_nonlinear_fn="tanh",
        char_cnn_pool_type="max",
        char_cnn_num_highway_layers=0,
        char_cnn_output_dim=-1,
        use_pretrained_weights=False,
        finetune_pretrained_weights=False,
        weights_file=None,
    ):
        super().__init__(dictionary)

        convolutions_params = literal_eval(char_cnn_params)
        self.char_cnn_encoder = char_encoder.CharCNNModel(
            dictionary,
            num_chars,
            embed_dim,
            convolutions_params,
            char_cnn_nonlinear_fn,
            char_cnn_pool_type,
            char_cnn_num_highway_layers,
            char_cnn_output_dim,
            use_pretrained_weights,
            finetune_pretrained_weights,
            weights_file,
        )

        self.embed_tokens = embed_tokens
        token_embed_dim = embed_tokens.embedding_dim
        self.word_layer_norm = nn.LayerNorm(token_embed_dim)

        char_embed_dim = (
            char_cnn_output_dim
            if char_cnn_output_dim != -1
            else sum(out_dim for (out_dim, _) in convolutions_params)
        )
        self.char_layer_norm = nn.LayerNorm(char_embed_dim)
        self.word_dim = char_embed_dim + token_embed_dim
        self.char_scale = math.sqrt(char_embed_dim / self.word_dim)
        self.word_scale = math.sqrt(token_embed_dim / self.word_dim)
        if self.word_dim != args.encoder_embed_dim:
            self.word_to_transformer_embed = fairseq_transformer.Linear(
                self.word_dim, args.encoder_embed_dim
            )

        self.dropout = args.dropout

        self.padding_idx = dictionary.pad()
        self.embed_positions = fairseq_transformer.PositionalEmbedding(
            1024,
            args.encoder_embed_dim,
            self.padding_idx,
            learned=args.encoder_learned_pos,
        )

        self.transformer_encoder_given_embeddings = TransformerEncoderGivenEmbeddings(
            args=args, proj_to_decoder=True
        )

        # Variable tracker
        self.tracker = VariableTracker()
        # Initialize adversarial mode
        self.set_gradient_tracking_mode(False)
        self.set_embed_noising_mode(False)

        # disables sorting and word-length thresholding if True
        # (enables ONNX tracing of length-sorted input with batch_size = 1)
        self.onnx_export_model = False

    def prepare_for_onnx_export_(self):
        self.onnx_export_model = True

    def set_gradient_tracking_mode(self, mode=True):
        """ This allows AdversarialTrainer to turn on retrain_grad when
        running adversarial example generation model."""
        self.tracker.reset()
        self.track_gradients = mode

    def set_embed_noising_mode(self, mode=True):
        """This allows adversarial trainer to turn on and off embedding noising
        layers. In regular training, this mode is off, and it is not included
        in forward pass.
        """
        self.embed_noising_mode = mode

    def forward(self, src_tokens, src_lengths, char_inds, word_lengths):
        self.tracker.reset()
        # char_inds has shape (batch_size, max_words_per_sent, max_word_len)
        bsz, seqlen, maxchars = char_inds.size()
        # char_cnn_encoder takes input (max_word_length, total_words)
        char_inds_flat = char_inds.view(-1, maxchars).t()
        # output (total_words, encoder_dim)
        char_cnn_output = self.char_cnn_encoder(char_inds_flat)
        x = char_cnn_output.view(bsz, seqlen, char_cnn_output.shape[-1])
        x = x.transpose(0, 1)  # (seqlen, bsz, char_cnn_output_dim)
        x = self.char_layer_norm(x)
        x = self.char_scale * x

        embedded_tokens = self.embed_tokens(src_tokens)
        # (seqlen, bsz, token_embed_dim)
        embedded_tokens = embedded_tokens.transpose(0, 1)
        embedded_tokens = self.word_layer_norm(embedded_tokens)
        embedded_tokens = self.word_scale * embedded_tokens
        x = torch.cat([x, embedded_tokens], dim=2)

        self.tracker.track(x, "token_embeddings", retain_grad=self.track_gradients)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.word_to_transformer_embed is not None:
            x = self.word_to_transformer_embed(x)
        positions = self.embed_positions(src_tokens)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # compute padding mask (B x T)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.transformer_encoder_given_embeddings(
            x=x, positions=positions, encoder_padding_mask=encoder_padding_mask
        )

        if self.onnx_export_model and encoder_padding_mask is None:
            encoder_padding_mask = torch.Tensor([]).type_as(src_tokens)

        return x, src_tokens, encoder_padding_mask

    def reorder_encoder_out(self, encoder_out, new_order):
        (x, src_tokens, encoder_padding_mask) = encoder_out
        if x is not None:
            x = x.index_select(1, new_order)
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.index_select(0, new_order)
        return (x, src_tokens, encoder_padding_mask)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if "encoder.embed_positions.weights" in state_dict:
                del state_dict["encoder.embed_positions.weights"]
            state_dict["encoder.embed_positions._float_tensor"] = torch.FloatTensor(1)
        return state_dict


@register_model_architecture("char_source_transformer", "char_source_transformer")
def base_architecture(args):
    # default architecture
    transformer.base_architecture(args)
    args.char_cnn_params = getattr(args, "char_cnn_params", "[(50, 1), (100,2)]")
    args.char_cnn_nonlinear_fn = getattr(args, "chr_cnn_nonlinear_fn", "relu")
    args.char_cnn_num_highway_layers = getattr(args, "char_cnn_num_highway_layers", "2")
