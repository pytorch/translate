#!/usr/bin/env python3

import logging
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from pytorch_translate import rnn  # noqa

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument(
        "--char-rnn",
        action="store_true",
        default=False,
        help="Assumes input is delimiter-separated character tokens "
        "(configurable with --word-delimiter option). Encodes word "
        "representations using bi-LSTM over char inputs.",
    )
    parser.add_argument(
        "--char-rnn-units",
        type=int,
        default=256,
        metavar="N",
        help=("Number of units for Character LSTM."),
    )

    parser.add_argument(
        "--char-rnn-layers",
        type=int,
        default=2,
        metavar="N",
        help=("Number of Character LSTM layers."),
    )
    parser.add_argument(
        "--word-delimiter", type=str, default="@SPACE", help=("Token separating words.")
    )


def set_arg_defaults(args):
    if hasattr(args, "char_rnn_params"):
        return args.char_rnn_params

    args.char_rnn_params = None
    char_rnn = getattr(args, "char_rnn", False)
    if char_rnn:
        args.char_rnn_params = {
            "char_rnn_units": args.char_rnn_units,
            "char_rnn_layers": args.char_rnn_layers,
            "word_delimiter": args.word_delimiter,
        }


class DelimiterSplit(nn.Module):
    """
    nn.Module which takes batched sequence input where the tokens are assumed
    to represent characters with a specified delimiter separating words, and
    returns the same indices split into words.

    Inputs:
    src_tokens (batch_size, max_length): character indices
    src_lengths (batch_size): lengths in total characters including delimiters

    Outputs:
    padded_char_inds (max_word_length, total_words)
    word_lenths (total_words,)
    words_per_sent (batch_size,)
    """

    def __init__(self, dictionary, word_delimiter="@SPACE"):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.word_delim_index = self.dictionary.index(word_delimiter)
        if self.word_delim_index == self.dictionary.unk():
            raise RuntimeError(
                f"Word delimiter {word_delimiter} not in source dictionary!"
            )

    def forward(self, src_tokens, src_lengths):
        words = []
        word_lengths = []
        words_per_sent = []
        src_tokens_numpy = src_tokens.cpu().numpy()
        for sentence_array in src_tokens_numpy:
            chars = []
            words_in_sentence = 0
            for idx in sentence_array:
                if idx == self.dictionary.pad():
                    continue
                elif idx == self.word_delim_index:
                    if len(chars) > 0:
                        word = torch.LongTensor(np.array(chars, dtype=np.int64))
                        words.append(word)
                        word_lengths.append(len(chars))
                        words_in_sentence += 1
                        chars = []
                        continue
                else:
                    chars.append(idx)
            if len(chars) > 0:
                word = torch.LongTensor(np.array(chars, dtype=np.int64))
                words.append(word)
                word_lengths.append(len(chars))
                words_in_sentence += 1
                chars = []
            words_per_sent.append(words_in_sentence)

        max_word_length = max(word_lengths)
        padded_char_inds = torch.LongTensor(max_word_length, len(words)).fill_(
            self.padding_idx
        )
        for idx, length in enumerate(word_lengths):
            padded_char_inds[:length, idx] = words[idx]

        # copy to GPU if necessary
        padded_char_inds = padded_char_inds.type_as(src_tokens)
        word_lengths = torch.LongTensor(word_lengths).type_as(src_lengths)
        words_per_sent = torch.LongTensor(words_per_sent).type_as(src_lengths)

        return padded_char_inds, word_lengths, words_per_sent


class CharRNN(nn.Module):
    """
    nn.Module to encode character sequences (with word delimiters) into
    vectors representing each word with bi-directional RNNS.
    """

    def __init__(
        self,
        dictionary,
        embed_dim,
        hidden_dim,
        num_layers,
        bidirectional=True,
        word_delimiter="@SPACE",
    ):
        super().__init__()

        self.word_split = DelimiterSplit(dictionary, word_delimiter)

        self.dictionary = dictionary
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_chars = rnn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
            freeze_embed=False,
        )

        self.bidirectional = bidirectional
        if self.bidirectional:
            assert hidden_dim % 2 == 0
        self.lstm_encoder = rnn.LSTMSequenceEncoder.LSTM(
            embed_dim,
            hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

    def forward(self, src_tokens, src_lengths):
        padded_char_inds, word_lengths, words_per_sent = self.word_split(
            src_tokens, src_lengths
        )

        # inputs to RNN must be in descending order of length
        sorted_word_lengths, word_length_order = torch.sort(
            word_lengths, descending=True
        )

        # shape: (max_word_len, total_words, embed_dim)
        char_rnn_input = self.embed_chars(padded_char_inds[:, word_length_order])

        packed_char_input = pack_padded_sequence(char_rnn_input, sorted_word_lengths)

        _, (h_last, _) = self.lstm_encoder(packed_char_input)

        # take last-layer output only (shape: (total_words, hidden_dim))
        if self.bidirectional:
            rnn_output = torch.cat((h_last[-2, :, :], h_last[-1, :, :]), dim=1)
        else:
            rnn_output = h_last[-1, :, :]

        # "unsort"
        _, inverted_word_length_order = torch.sort(word_length_order)
        unsorted_rnn_output = rnn_output[inverted_word_length_order, :]

        # (max_words_per_sent, batch_size, word_rep_dim)
        output = torch.FloatTensor(
            int(words_per_sent.max()), words_per_sent.shape[0], rnn_output.size(1)
        ).type_as(
            rnn_output
        )

        sent_end_indices = words_per_sent.cumsum(0)
        for sent_index in range(words_per_sent.shape[0]):
            start = 0 if sent_index == 0 else sent_end_indices[sent_index - 1]
            end = sent_end_indices[sent_index]
            output[: words_per_sent[sent_index], sent_index, :] = unsorted_rnn_output[
                start:end, :
            ]

        return output, words_per_sent
