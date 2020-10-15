#!/usr/bin/env python3

import logging
from ast import literal_eval

import torch
import torch.nn.functional as F
from fairseq.models import FairseqEncoder, register_model, register_model_architecture
from pytorch_translate import char_encoder, model_constants, rnn, utils, vocab_constants
from pytorch_translate.common_layers import Embedding, VariableTracker
from pytorch_translate.data.dictionary import TAGS


logger = logging.getLogger(__name__)


@register_model("char_source")
class CharSourceModel(rnn.RNNModel):
    def __init__(self, task, encoder, decoder):
        super().__init__(task, encoder, decoder)

    @staticmethod
    def add_args(parser):
        rnn.RNNModel.add_args(parser)
        char_encoder.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        src_dict, dst_dict = task.source_dictionary, task.target_dictionary
        base_architecture(args)

        assert args.sequence_lstm, "CharRNNModel only supports sequence_lstm"
        assert args.cell_type == "lstm", "CharRNNModel only supports cell_type lstm"

        assert hasattr(args, "char_source_dict_size"), (
            "args.char_source_dict_size required. "
            "should be set by load_binarized_dataset()"
        )

        if hasattr(args, "char_cnn_params"):
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
                src_dict,
                num_chars=num_chars,
                unk_only_char_encoding=args.unk_only_char_encoding,
                embed_dim=args.char_embed_dim,
                token_embed_dim=args.encoder_embed_dim,
                freeze_embed=args.encoder_freeze_embed,
                normalize_embed=args.encoder_normalize_embed,
                char_cnn_params=args.char_cnn_params,
                char_cnn_nonlinear_fn=args.char_cnn_nonlinear_fn,
                char_cnn_num_highway_layers=args.char_cnn_num_highway_layers,
                char_cnn_output_dim=getattr(args, "char_cnn_output_dim", -1),
                num_layers=args.encoder_layers,
                hidden_dim=args.encoder_hidden_dim,
                dropout_in=args.encoder_dropout_in,
                dropout_out=args.encoder_dropout_out,
                residual_level=args.residual_level,
                bidirectional=bool(args.encoder_bidirectional),
                use_pretrained_weights=getattr(args, "use_pretrained_weights", False),
                finetune_pretrained_weights=getattr(
                    args, "finetune_pretrained_weights", False
                ),
                weights_file=getattr(args, "pretrained_weights_file", ""),
            )
        else:
            assert (
                args.unk_only_char_encoding is False
            ), "unk_only_char_encoding should be False when using CharRNNEncoder"

            encoder = CharRNNEncoder(
                src_dict,
                num_chars=args.char_source_dict_size,
                char_embed_dim=args.char_embed_dim,
                token_embed_dim=args.encoder_embed_dim,
                normalize_embed=args.encoder_normalize_embed,
                char_rnn_units=args.char_rnn_units,
                char_rnn_layers=args.char_rnn_layers,
                num_layers=args.encoder_layers,
                hidden_dim=args.encoder_hidden_dim,
                dropout_in=args.encoder_dropout_in,
                dropout_out=args.encoder_dropout_out,
                residual_level=args.residual_level,
                bidirectional=bool(args.encoder_bidirectional),
            )

        decoder_embed_tokens = Embedding(
            num_embeddings=len(dst_dict),
            embedding_dim=args.decoder_embed_dim,
            padding_idx=dst_dict.pad(),
            freeze_embed=args.decoder_freeze_embed,
        )

        utils.load_embedding(
            embedding=decoder_embed_tokens,
            dictionary=dst_dict,
            pretrained_embed=args.decoder_pretrained_embed,
        )
        decoder = rnn.RNNDecoder(
            src_dict=src_dict,
            dst_dict=dst_dict,
            embed_tokens=decoder_embed_tokens,
            vocab_reduction_params=args.vocab_reduction_params,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            out_embed_dim=args.decoder_out_embed_dim,
            cell_type=args.cell_type,
            num_layers=args.decoder_layers,
            hidden_dim=args.decoder_hidden_dim,
            attention_type=args.attention_type,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            residual_level=args.residual_level,
            averaging_encoder=args.averaging_encoder,
        )
        return cls(task, encoder, decoder)

    def forward(
        self, src_tokens, src_lengths, char_inds, word_lengths, prev_output_tokens
    ):
        """
        Overriding FairseqEncoderDecoderModel.forward() due to different encoder
        inputs.
        """
        encoder_out = self.encoder(src_tokens, src_lengths, char_inds, word_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out


class CharRNNEncoder(FairseqEncoder):
    """
    RNN encoder encoding each word via a bidirectional LSTM over character
    embeddings to obtain word representations, and then an LSTM (optionally
    bidirectional in first layer) to combine word representations.

    Uses nn.LSTM for cuDNN support / ONNX exportability.
    """

    def __init__(
        self,
        dictionary,
        num_chars,
        char_embed_dim,
        token_embed_dim,
        normalize_embed,
        char_rnn_units,
        char_rnn_layers,
        hidden_dim,
        num_layers,
        dropout_in,
        dropout_out,
        residual_level,
        bidirectional,
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in

        self.embed_chars = char_encoder.CharRNNModel(
            dictionary=dictionary,
            num_chars=num_chars,
            char_embed_dim=char_embed_dim,
            char_rnn_units=char_rnn_units,
            char_rnn_layers=char_rnn_layers,
        )

        self.embed_tokens = None
        if token_embed_dim > 0:
            self.embed_tokens = rnn.Embedding(
                num_embeddings=len(dictionary),
                embedding_dim=token_embed_dim,
                padding_idx=dictionary.pad(),
                freeze_embed=False,
                normalize_embed=normalize_embed,
            )

        self.word_dim = char_rnn_units + token_embed_dim

        self.bilstm = rnn.BiLSTM(
            num_layers=num_layers,
            bidirectional=bidirectional,
            embed_dim=self.word_dim,
            hidden_dim=hidden_dim,
            dropout=dropout_out,
            residual_level=residual_level,
        )

        # disables sorting and word-length thresholding if True
        # (enables ONNX tracing of length-sorted input with batch_size = 1)
        self.onnx_export_model = False

    def forward(self, src_tokens, src_lengths, char_inds, word_lengths):
        x = self.embed_chars(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            char_inds=char_inds,
            word_lengths=word_lengths,
        )

        if self.embed_tokens is not None:
            embedded_tokens = self.embed_tokens(src_tokens)

            # (seqlen, bsz, token_embed_dim)
            embedded_tokens = embedded_tokens.transpose(0, 1)

            # (seqlen, bsz, total_word_embed_dim)
            x = torch.cat([x, embedded_tokens], dim=2)

        if self.dropout_in != 0:
            x = F.dropout(x, p=self.dropout_in, training=self.training)

        embedded_words = x

        unpacked_output, final_hiddens, final_cells = self.bilstm(
            embeddings=x, lengths=src_lengths
        )

        return (
            unpacked_output,
            final_hiddens,
            final_cells,
            src_lengths,
            src_tokens,
            embedded_words,
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder all outputs according to new_order."""
        return rnn.reorder_encoder_output(encoder_out, new_order)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class CharCNNEncoder(FairseqEncoder):
    """
    Character-level CNN encoder to generate word representations, as input to
    RNN encoder.
    """

    def __init__(
        self,
        dictionary,
        num_chars=50,
        unk_only_char_encoding=False,
        embed_dim=32,
        token_embed_dim=256,
        freeze_embed=False,
        normalize_embed=False,
        char_cnn_params="[(128, 3), (128, 5)]",
        char_cnn_nonlinear_fn="tanh",
        char_cnn_num_highway_layers=0,
        char_cnn_output_dim=-1,
        hidden_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        residual_level=None,
        bidirectional=False,
        use_pretrained_weights=False,
        finetune_pretrained_weights=False,
        weights_file=None,
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in

        convolutions_params = literal_eval(char_cnn_params)
        self.char_cnn_encoder = char_encoder.CharCNNModel(
            dictionary,
            num_chars,
            embed_dim,
            convolutions_params,
            char_cnn_nonlinear_fn,
            char_cnn_num_highway_layers,
            char_cnn_output_dim,
            use_pretrained_weights,
            finetune_pretrained_weights,
            weights_file,
        )

        self.embed_tokens = None
        num_tokens = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.unk_idx = dictionary.unk()
        if token_embed_dim > 0:
            self.embed_tokens = rnn.Embedding(
                num_embeddings=num_tokens,
                embedding_dim=token_embed_dim,
                padding_idx=self.padding_idx,
                freeze_embed=freeze_embed,
                normalize_embed=normalize_embed,
            )
        self.word_dim = (
            char_cnn_output_dim
            if char_cnn_output_dim != -1
            else sum(out_dim for (out_dim, _) in convolutions_params)
        )
        self.token_embed_dim = token_embed_dim

        self.unk_only_char_encoding = unk_only_char_encoding
        if self.unk_only_char_encoding:
            assert (
                char_cnn_output_dim == token_embed_dim
            ), "char_cnn_output_dim (%d) must equal to token_embed_dim (%d)" % (
                char_cnn_output_dim,
                token_embed_dim,
            )
            self.word_dim = token_embed_dim
        else:
            self.word_dim = self.word_dim + token_embed_dim

        self.bilstm = rnn.BiLSTM(
            num_layers=num_layers,
            bidirectional=bidirectional,
            embed_dim=self.word_dim,
            hidden_dim=hidden_dim,
            dropout=dropout_out,
            residual_level=residual_level,
        )

        # Variable tracker
        self.tracker = VariableTracker()
        # Initialize adversarial mode
        self.set_gradient_tracking_mode(False)
        self.set_embed_noising_mode(False)

    def set_gradient_tracking_mode(self, mode=True):
        """This allows AdversarialTrainer to turn on retrain_grad when
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
        char_inds_flat = char_inds.view(-1, maxchars)  # .t()
        # output (total_words, encoder_dim)
        if self.unk_only_char_encoding:
            assert (
                self.embed_tokens is not None
            ), "token_embed_dim should > 0 when unk_only_char_encoding is true!"

            unk_masks = (src_tokens == self.unk_idx).view(-1)
            unk_indices = torch.nonzero(unk_masks).squeeze()
            if unk_indices.dim() > 0 and unk_indices.size(0) > 0:
                char_inds_flat = torch.index_select(char_inds_flat, 0, unk_indices)
                char_inds_flat = char_inds_flat.view(-1, maxchars)
            else:
                char_inds_flat = None

        if char_inds_flat is not None:
            # (bsz * seqlen, encoder_dim)
            char_cnn_output = self.char_cnn_encoder(char_inds_flat.t())
            x = char_cnn_output
        else:  # charCNN is not needed
            x = None

        if self.embed_tokens is not None:
            # (bsz, seqlen, token_embed_dim)
            embedded_tokens = self.embed_tokens(src_tokens)
            # (bsz * seqlen, token_embed_dim)
            embedded_tokens = embedded_tokens.view(-1, self.token_embed_dim)
            if self.unk_only_char_encoding:  # charCNN for UNK words only
                if x is not None:
                    x = embedded_tokens.index_copy(0, unk_indices, x)
                else:  # no UNK, so charCNN is not needed
                    x = embedded_tokens
            else:  # charCNN for all words
                x = torch.cat([x, embedded_tokens], dim=1)

        # (bsz, seqlen, x.shape[-1])
        x = x.view(bsz, seqlen, x.shape[-1])
        # (seqlen, bsz, x.shape[-1])
        x = x.transpose(0, 1)

        self.tracker.track(x, "token_embeddings", retain_grad=self.track_gradients)
        if self.dropout_in != 0:
            x = F.dropout(x, p=self.dropout_in, training=self.training)
        embedded_words = x

        unpacked_output, final_hiddens, final_cells = self.bilstm(
            embeddings=x, lengths=src_lengths
        )

        return (
            unpacked_output,
            final_hiddens,
            final_cells,
            src_lengths,
            src_tokens,
            embedded_words,
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder all outputs according to new_order."""
        return rnn.reorder_encoder_output(encoder_out, new_order)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


def verify_pretrain_params(args):
    """
    Function which verifies the model params in case we are using a pretrained
    model like ELMo. This is needed because the model params must match up
    with the weights that are being loaded.
    """
    assert (
        args.embed_bytes
    ), "To use pretrained weights, embed_bytes must be set to True."

    assert (
        args.char_cnn_nonlinear_fn == model_constants.PRETRAINED_CHAR_CNN_NONLINEAR_FN
    ), "To use pretrained weights, the non linearity used should be relu."

    assert (
        args.char_embed_dim == model_constants.PRETRAINED_CHAR_EMBED_DIM
    ), "To use pretrained weights char_embed_dim should be set to 16."

    assert (
        args.char_cnn_output_dim == model_constants.PRETRAINED_CHAR_CNN_OUTPUT_DIM
    ), "To use pretrained weights, the output dim of the CNN layer should be 512."
    assert (
        literal_eval(args.char_cnn_params) == model_constants.PRETRAINED_CHAR_CNN_PARAMS
    ), "CNN Params don't match with the ones needed for loading pretrained weights"


@register_model_architecture("char_source", "char_source")
def base_architecture(args):
    # default architecture
    rnn.base_architecture(args)
    args.char_rnn_units = getattr(args, "char_embed_dim", 128)
    args.char_rnn_units = getattr(args, "char_rnn_units", 256)
    args.char_rnn_layers = getattr(args, "char_rnn_layers", 1)
