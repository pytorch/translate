from torch.autograd import Variable

from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqModel,
)
from fbtranslate import vocab_reduction
from fbtranslate.rnn import (
    torch_find,
    LSTMSequenceEncoder,
    RNNEncoder,
    RNNDecoder,
)
from .word_predictor import WordPredictor


class FairseqWordPredictionModel(FairseqModel):
    def __init__(self, encoder, decoder, predictor):
        super().__init__(encoder, decoder)
        self.predictor = predictor

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_output = self.encoder(src_tokens, src_lengths)
        pred_output = self.predictor(encoder_output)
        decoder_output = self.decoder(prev_output_tokens, encoder_output)
        return pred_output, decoder_output

    def get_predictor_normalized_probs(self, pred_output, log_probs):
        return self.predictor.get_normalized_probs(pred_output, log_probs)

    def get_target_words(self, sample):
        return sample['target']


@register_model('rnn_wp')
class RNNWordPredictionModel(FairseqWordPredictionModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--dropout',
            default=0.1,
            type=float,
            metavar='D',
            help='dropout probability',
        )
        parser.add_argument(
            '--encoder-embed-dim',
            type=int,
            metavar='N',
            help='encoder embedding dimension',
        )
        parser.add_argument(
            '--encoder-freeze-embed',
            default=False,
            action='store_true',
            help=('whether to freeze the encoder embedding or allow it to be '
                  'updated during training'),
        )
        parser.add_argument(
            '--encoder-hidden-dim',
            type=int,
            metavar='N',
            help='encoder cell num units',
        )
        parser.add_argument(
            '--encoder-layers',
            type=int,
            metavar='N',
            help='number of encoder layers',
        )
        parser.add_argument(
            '--encoder-bidirectional',
            action='store_true',
            help='whether the first layer is bidirectional or not',
        )
        parser.add_argument(
            '--averaging-encoder',
            default=False,
            action='store_true',
            help=(
                'whether use mean encoder hidden states as decoder initial '
                'states or not'
            ),
        )
        parser.add_argument(
            '--decoder-embed-dim',
            type=int,
            metavar='N',
            help='decoder embedding dimension',
        )
        parser.add_argument(
            '--decoder-freeze-embed',
            default=False,
            action='store_true',
            help=('whether to freeze the encoder embedding or allow it to be '
                  'updated during training'),
        )
        parser.add_argument(
            '--decoder-hidden-dim',
            type=int,
            metavar='N',
            help='decoder cell num units',
        )
        parser.add_argument(
            '--decoder-layers',
            type=int,
            metavar='N',
            help='number of decoder layers',
        )
        parser.add_argument(
            '--decoder-out-embed-dim',
            type=int,
            metavar='N',
            help='decoder output embedding dimension',
        )
        parser.add_argument(
            '--attention-type',
            type=str,
            metavar='EXPR',
            help='decoder attention, defaults to dot',
        )
        parser.add_argument(
            '--residual-level',
            default=None,
            type=int,
            help=(
                'First layer where to apply a residual connection. '
                'The value should be greater than 0 and smaller than the number of '
                'layers.'
            ),
        )
        parser.add_argument(
            '--cell-type',
            default='lstm',
            type=str,
            metavar='EXPR',
            help='cell type, defaults to lstm, values:lstm, milstm, layer_norm_lstm',
        )

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument(
            '--encoder-dropout-in',
            type=float,
            metavar='D',
            help='dropout probability for encoder input embedding',
        )
        parser.add_argument(
            '--encoder-dropout-out',
            type=float,
            metavar='D',
            help='dropout probability for encoder output',
        )
        parser.add_argument(
            '--decoder-dropout-in',
            type=float,
            metavar='D',
            help='dropout probability for decoder input embedding',
        )
        parser.add_argument(
            '--decoder-dropout-out',
            type=float,
            metavar='D',
            help='dropout probability for decoder output',
        )
        parser.add_argument(
            '--sequence-lstm',
            action='store_true',
            help='use nn.LSTM implementation for encoder',
        )
        # new arg
        parser.add_argument(
            '--predictor-hidden-dim',
            type=int,
            metavar='N',
            help='word predictor num units',
        )

        # Args for vocab reduction
        vocab_reduction.add_args(parser)

    @classmethod
    def build_model(cls, args, src_dict, dst_dict):
        """Build a new model instance."""
        base_architecture_wp(args)
        if args.sequence_lstm:
            encoder_class = LSTMSequenceEncoder
        else:
            encoder_class = RNNEncoder
        encoder = encoder_class(
            src_dict,
            embed_dim=args.encoder_embed_dim,
            freeze_embed=args.encoder_freeze_embed,
            cell_type=args.cell_type,
            num_layers=args.encoder_layers,
            hidden_dim=args.encoder_hidden_dim,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            residual_level=args.residual_level,
            bidirectional=bool(args.encoder_bidirectional),
        )
        decoder = RNNDecoder(
            src_dict=src_dict,
            dst_dict=dst_dict,
            vocab_reduction_params=args.vocab_reduction_params,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            freeze_embed=args.decoder_freeze_embed,
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
        predictor = WordPredictor(
            args.encoder_hidden_dim, args.predictor_hidden_dim, len(dst_dict)
        )
        return cls(encoder, decoder, predictor)

    def get_targets(self, sample, net_output):
        targets = sample['target'].view(-1)
        possible_translation_tokens = net_output[-1]
        if possible_translation_tokens is not None:
            targets = Variable(torch_find(
                possible_translation_tokens.data,
                targets.data,
                len(self.dst_dict),
            ))
        return targets


@register_model_architecture('rnn_wp', 'rnn_wp')
def base_architecture_wp(args):
    # default architecture
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 512)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 512)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.attention_type = getattr(args, 'attention_type', 'dot')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.averaging_encoder = getattr(args, 'averaging_encoder', False)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.cell_type = getattr(args, 'cell_type', 'lstm')
    vocab_reduction.set_arg_defaults(args)
    args.sequence_lstm = getattr(args, 'sequence_lstm', False)
    args.predictor_hidden_dim = getattr(args, 'predictor_hidden_dim', 512)
