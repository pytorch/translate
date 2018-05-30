#!/usr/bin/env python3

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import ConvTBC
from pytorch_translate import common_layers  # noqa


class HighwayLayer(nn.Module):

    def __init__(
        self,
        input_dim,
        transform_activation=F.relu,
        gate_activation=F.softmax,
        # Srivastava et al. (2015) recommend initializing bT to a negative
        # value, in order to militate the initial behavior towards carry.
        # We initialized bT to a small interval around −2
        gate_bias=-2,
    ):
        super().__init__()
        self.highway_transform_activation = transform_activation
        self.highway_gate_activation = gate_activation
        self.highway_transform = nn.Linear(input_dim, input_dim)
        self.highway_gate = nn.Linear(input_dim, input_dim)
        self.highway_gate.bias.data.fill_(gate_bias)

    def forward(self, x):
        transform_output = self.highway_transform_activation(self.highway_transform(x))
        gate_output = self.highway_gate_activation(self.highway_gate(x))

        transformation_part = torch.mul(transform_output, gate_output)
        carry_part = torch.mul((1 - gate_output), x)
        return torch.add(transformation_part, carry_part)


class CharEmbModel(nn.Module):
    """
    A Conv network to generate word embedding from character embeddings, from
    Character-Aware Neural Language Models, https://arxiv.org/abs/1508.06615.

    Components include convolutional filters, max pooling, and
    optional highway network.

    Input dim: [total_num_words_per_batch, max_word_width]
    Such input results from a serial of transformation outside of the model:
    - after yoda.core.generic.transliteration_base:
        'h e l l o @SPACE @DIGITS @SPACE ! @SPACE w e ' r e @SPACE h e r e @SPACE'
    - after numberization by VocabProcessor:
        '8 5 12 12 15 @SPACE 43 @SPACE 44 @SPACE 23 5 45 18 5 @SPACE ...'
    - after character batching logic, i.e. breaking into words by @SPACE:
      [total_num_words_per_batch, max_word_width]
        'BOW 8 5 12 12 15 EOW
         BOW 43 EOW PAD PAD PAD
         BOW 44 EOW PAD PAD PAD
         BOW 23 5 45 18 5 EOW
         ....
        '
    To combine those to generate word embeddings per sentence, need to remember
    number of words in each sentence before breaking into words.
    """

    def __init__(
        self,
        dictionary,
        char_embed_dim=32,
        word_embed_dim=512,
        convolutions=((128, 3), (128, 5)),
        dropout=0.1,
        num_highway_layers=0,
        preserve_word=True,
    ):
        super().__init__()
        self.dictionary = dictionary
        vocab_size = len(self.dictionary)
        self.embed_char_tokens = nn.Embedding(vocab_size, char_embed_dim)
        in_channels = convolutions[0][0]
        self.dropout = dropout
        self.convolutions = nn.ModuleList(
            [
                ConvTBC(in_channels, out_channels * 2, kernel_size)
                for (out_channels, kernel_size) in convolutions
            ]
        )

        self.fc_input = common_layers.Linear(char_embed_dim, in_channels)
        conv_output_dim = sum(out_dim for (out_dim, _) in convolutions)
        self.fc_output = common_layers.Linear(conv_output_dim, word_embed_dim)

        self.highway_layers = nn.ModuleList(
            [HighwayLayer(conv_output_dim)] * num_highway_layers
        )

        self.preserve_word = preserve_word

    def _prepare_char_batch(self, src_tokens, left_padded=True):
        """Transform sentence batch into word batch, given inputs like
        src_tokens: [batch_size, max_seq_len],
            Padded sequence of numberized characters, with words separated by
            self.dictionary.word_delim_index.
        """
        # Split sentence into words and flatten them
        word_delim_mask = src_tokens.eq(self.dictionary.word_delim_index)
        splitted_words = [
            np.split(src_sent, delim_mask.nonzero()[0])
            for src_sent, delim_mask in zip(src_tokens.numpy(), word_delim_mask.numpy())
        ]

        # Remove sentence pads
        for sent in splitted_words:
            if left_padded:
                sent[0] = np.array(
                    [
                        char_token
                        for char_token in sent[0]
                        if char_token != self.dictionary.pad_index
                    ]
                )
            else:
                sent[-1] = np.array(
                    [
                        char_token
                        for char_token in sent[-1]
                        if char_token != self.dictionary.pad_index
                    ]
                )
        # Flat all words within each sentence,
        # use src_lengths to restore senteence later on
        flattened_words = np.array([sent for sent in splitted_words]).flatten()
        # Remove word delim token
        pure_words = [
            word if word[0] != self.dictionary.word_delim_index else word[1:]
            for word in flattened_words
        ]

        # Add BOW and EOW, and pad each word to max word length within this batch
        max_word_len = np.max([len(word) for word in pure_words])
        words_batch = np.asarray(
            [
                np.pad(
                    np.hstack(
                        ([self.dictionary.bow_index], word, [self.dictionary.eow_index])
                    ),
                    (0, max_word_len - len(word)),
                    "constant",
                    constant_values=self.dictionary.pad_index,
                )
                for word in pure_words
            ]
        )

        return torch.LongTensor(words_batch)

    def forward(self, src_tokens, src_lengths, left_padded=True):
        # src_char_tokens are the desired batch, but could be batch of words
        # or batch of sentences, depends on:
        # if preserve_word: break sentence into words apply conv-pool-highway
        # to generate word embeddings.
        # else: apply conv-pool-highway directly on sentence batch, generate
        # segment embeddings per sentence
        if self.preserve_word:
            src_char_tokens = self._prepare_char_batch(src_tokens, left_padded)
        else:
            src_char_tokens = src_tokens

        x = self.embed_char_tokens(src_char_tokens)
        x = F.dropout(x, p=self.dropout)
        # project to the size of convolution
        x = self.fc_input(x)

        # [batch_size, time_step, channel_size] -> [time_step, batch_size, channel_size}
        x = x.transpose(0, 1)
        # temporal convoluation across characters
        kernel_outputs = []
        for conv in self.convolutions:
            conv_output = conv(x)
            # append (filtered_seq_len, batch_size, output_channel_size)
            kernel_outputs.append(F.glu(conv_output, dim=2))

        # pooling over the time dimension, return [(pooled value, max_index)]
        pools = [torch.max(conv, 0)[0] for conv in kernel_outputs]

        # [batch_size, sum(output_channel_dim)]
        x = torch.cat([p for p in pools], 1)

        # highway layer transformation
        for highway_layer in self.highway_layers:
            x = highway_layer(x)

        # project from output_channel_dim to the size of encoder input dim
        encoder_input = self.fc_output(x)

        # [batch_size, seq_len, word_emb_dim]
        # TODO: add pads if variable seq_len
        if self.preserve_word:
            splits = torch.cumsum(torch.Tensor([0] + src_lengths), dim=0)[:-1]
            split_along_dim = 0
            reshaped_encoder_input = torch.cat(
                [
                    encoder_input.narrow(
                        int(split_along_dim), int(start), int(length)
                    ).unsqueeze(0)
                    for start, length in zip(splits, src_lengths)
                ]
            )
            return reshaped_encoder_input

        return encoder_input
