#!/usr/bin/env python3

import math
from ast import literal_eval
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture
from pytorch_translate import (
    char_encoder,
    char_source_hybrid,
    char_source_model,
    hybrid_transformer_rnn,
    transformer as pytorch_translate_transformer,
    vocab_constants,
)
from pytorch_translate.data.dictionary import TAGS
from pytorch_translate.utils import maybe_cuda


@register_model("char_aware_hybrid")
class CharAwareHybridModel(char_source_hybrid.CharSourceHybridModel):
    """
    An architecture combining hybrid Transformer/RNN with character-based
    inputs (token embeddings created via character-input CNN) and outputs.
    This model is very similar to https://arxiv.org/pdf/1809.02223.pdf.
    """

    def __init__(self, task, encoder, decoder):
        super().__init__(task, encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        src_dict, dst_dict = task.source_dictionary, task.target_dictionary
        base_architecture(args)

        assert hasattr(args, "char_source_dict_size"), (
            "args.char_source_dict_size required. "
            "should be set by load_binarized_dataset()"
        )
        assert hasattr(args, "char_target_dict_size"), (
            "args.char_target_dict_size required. "
            "should be set by load_binarized_dataset()"
        )

        assert hasattr(
            args, "char_cnn_params"
        ), "Only char CNN is supported for the char encoder hybrid model"

        args.embed_bytes = getattr(args, "embed_bytes", False)

        # In case use_pretrained_weights is true, verify the model params
        # are correctly set
        if args.embed_bytes and getattr(args, "use_pretrained_weights", False):
            char_source_model.verify_pretrain_params(args)

        encoder = char_source_hybrid.CharSourceHybridModel.build_encoder(
            args=args, src_dict=src_dict
        )
        decoder = CharAwareHybridModel.build_decoder(
            args=args, src_dict=src_dict, dst_dict=dst_dict
        )

        return cls(task, encoder, decoder)

    def forward(
        self,
        src_tokens,
        src_lengths,
        char_inds,
        word_lengths,
        prev_output_tokens,
        prev_output_chars,
        prev_output_word_lengths=None,
    ):
        encoder_out = self.encoder(src_tokens, src_lengths, char_inds, word_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            prev_output_chars=prev_output_chars,
        )
        return decoder_out

    @classmethod
    def build_decoder(cls, args, src_dict, dst_dict):
        # If we embed bytes then the number of indices is fixed and does not
        # depend on the dictionary
        if args.embed_bytes:
            num_chars = vocab_constants.NUM_BYTE_INDICES + TAGS.__len__() + 1
        else:
            num_chars = args.char_target_dict_size

        decoder_embed_tokens = pytorch_translate_transformer.build_embedding(
            dictionary=dst_dict,
            embed_dim=args.decoder_embed_dim,
            path=args.decoder_pretrained_embed,
            freeze=args.decoder_freeze_embed,
        )
        return CharAwareHybridRNNDecoder(
            args,
            src_dict=src_dict,
            dst_dict=dst_dict,
            embed_tokens=decoder_embed_tokens,
            num_chars=num_chars,
            char_embed_dim=args.char_embed_dim,
            char_cnn_params=args.char_cnn_params,
            char_cnn_nonlinear_fn=args.char_cnn_nonlinear_fn,
            char_cnn_num_highway_layers=args.char_cnn_num_highway_layers,
            use_pretrained_weights=False,
            finetune_pretrained_weights=False,
        )


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
            num_highway_layers=char_cnn_num_highway_layers,
            # char_cnn_output_dim should match the word embedding dimension.
            char_cnn_output_dim=embed_tokens.embedding_dim,
            use_pretrained_weights=use_pretrained_weights,
            finetune_pretrained_weights=finetune_pretrained_weights,
        )
        self.char_layer_norm = nn.LayerNorm(embed_tokens.embedding_dim)

        # By default (before training ends), character representations are
        # not precomputed. After precomputation, this value should be used in place of
        # the two embeddings.
        self._is_precomputed = False
        self.combined_word_char_embed = nn.Embedding(
            embed_tokens.num_embeddings, embed_tokens.embedding_dim
        )

    def _get_char_cnn_output(self, char_inds):
        if char_inds.dim() == 2:
            char_inds = char_inds.unsqueeze(1)
        bsz, seqlen, maxchars = char_inds.size()

        # char_cnn_encoder takes input (max_word_length, total_words)
        char_inds_flat = char_inds.view(-1, maxchars).t()
        # output (total_words, encoder_dim)
        char_cnn_output = self.char_cnn_encoder(char_inds_flat)

        char_cnn_output = char_cnn_output.view(bsz, seqlen, char_cnn_output.shape[-1])
        # (seqlen, bsz, char_cnn_output_dim)
        char_cnn_output = char_cnn_output.transpose(0, 1)
        char_cnn_output = self.char_layer_norm(char_cnn_output)
        return char_cnn_output

    def _embed_prev_outputs(
        self, prev_output_tokens, incremental_state=None, prev_output_chars=None
    ):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if prev_output_chars is not None:
                prev_output_chars = prev_output_chars[:, -1:, :].squeeze(1)

        combined_embed = self._combined_word_char_embed(
            prev_output_tokens=prev_output_tokens, prev_output_chars=prev_output_chars
        )
        return combined_embed, prev_output_tokens

    def _combined_word_char_embed(self, prev_output_tokens, prev_output_chars):
        """
        If the embeddings are precomputed for character compositions (this holds
        in inference), use the cached embeddings, otherwise compute it.
        """
        if self._is_precomputed:
            combined_embedding = (
                self.combined_word_char_embed(prev_output_tokens)
                .squeeze(1)
                .unsqueeze(0)
            )
        else:
            x = self.embed_tokens(prev_output_tokens)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            char_cnn_output = self._get_char_cnn_output(prev_output_chars)
            combined_embedding = x + char_cnn_output
        return combined_embedding

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state=None,
        possible_translation_tokens=None,
        timestep=None,
        prev_output_chars=None,
    ):
        """
        The assumption is that during inference, the word embedding values are
        summed with their corresponding character representations. Thus the model
        will look like the same as a word-based decoder.
        """
        if self.training:
            x, prev_output_tokens = self._embed_prev_outputs(
                prev_output_tokens=prev_output_tokens,
                incremental_state=incremental_state,
                prev_output_chars=prev_output_chars,
            )
        else:
            x, prev_output_tokens = super()._embed_prev_outputs(
                prev_output_tokens=prev_output_tokens,
                incremental_state=incremental_state,
            )
        return self._forward_given_embeddings(
            embed_out=x,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            possible_translation_tokens=possible_translation_tokens,
            timestep=timestep,
        )

    def precompute_char_representations(
        self, char_dict, embed_bytes=False, batch_size=5000
    ):
        """
        Precomputes the embeddings from character CNNs. Then adds that to the
        word embeddings.
        Args:
            batch_size: maximum number of words in one batch
        """
        character_list = self._char_list_from_dict(
            char_dict=char_dict, embed_bytes=embed_bytes
        )
        all_idx = maybe_cuda(
            torch.LongTensor([i for i in range(self.embed_tokens.num_embeddings)])
        )
        word_embeds = self.embed_tokens(all_idx)
        num_minibatches = math.ceil(len(character_list) / batch_size)
        for i in range(num_minibatches):
            character_sublist = character_list[
                i * batch_size : min((i + 1) * batch_size, len(character_list))
            ]
            max_word_len = max(len(chars) for chars in character_sublist)
            char_inds = (
                torch.Tensor(len(character_sublist), max_word_len)
                .long()
                .fill_(char_dict.pad_index)
            )

            for j, chars in enumerate(character_sublist):
                char_inds[j, : len(chars)] = torch.LongTensor(chars)

            char_cnn_output = self._get_char_cnn_output(maybe_cuda(char_inds))

            # Filling in the precomputed embedding values.
            index_offset = i * batch_size
            for j in range(char_cnn_output.size()[1]):
                cur_idx = j + index_offset
                self.combined_word_char_embed.weight[cur_idx] = (
                    char_cnn_output[0, j, :] + word_embeds[cur_idx]
                )

        self._is_precomputed = True
        self.combined_word_char_embed.weight.detach()

    def _char_list_from_dict(self, char_dict, embed_bytes=False) -> List[List[int]]:
        """
        From self.word_dict, extracts all character sequneces, and convert
        them to their corresponding list of characters.
        """
        character_list = []
        for word_index, word in enumerate(self.dictionary.symbols):
            character_list.append(
                self._char_list_for_word(
                    word_index=word_index,
                    word=word,
                    char_dict=char_dict,
                    embed_bytes=embed_bytes,
                )
            )
        return character_list

    def _char_list_for_word(
        self, word_index: int, word: str, char_dict, embed_bytes=False
    ) -> List[int]:
        """
        Extracts character
        For special words except pad, we put eos, because we actually
        do not need their character sequences.
        """
        if word_index == self.dictionary.pad_index:
            char_inds = [char_dict.pad_index]
        elif word_index < self.dictionary.nspecial:
            char_inds = [char_dict.eos_index]
        else:
            if embed_bytes:
                # The byte_id needs to be incremented by 1 to account for the
                # padding id (0) in the embedding table
                char_inds = (
                    [vocab_constants.NUM_BYTE_INDICES + TAGS.index(word) + 1]
                    if word in TAGS
                    else [byte_id + 1 for byte_id in word.encode("utf8", "ignore")]
                )
            else:
                chars = [word] if word in TAGS else list(word)
                char_inds = [char_dict.index(c) for c in chars]
        return char_inds


@register_model_architecture("char_aware_hybrid", "char_aware_hybrid")
def base_architecture(args):
    # default architecture
    hybrid_transformer_rnn.base_architecture(args)
    args.char_cnn_params = getattr(args, "char_cnn_params", "[(50, 1), (100,2)]")
    args.char_cnn_nonlinear_fn = getattr(args, "chr_cnn_nonlinear_fn", "relu")
    args.char_cnn_num_highway_layers = getattr(args, "char_cnn_num_highway_layers", "2")
