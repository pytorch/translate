#!/usr/bin/env python3

import torch
import torch.nn as nn
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder
from pytorch_translate import data as pytorch_translate_data
from pytorch_translate import utils


class MultilingualEncoder(FairseqEncoder):
    """Multilingual encoder.

    This encoder consists of n separate encoders. A language token ID at the
    begin of the source sentence selects the encoder to use.
    """

    def __init__(self, dictionary, encoders, hidden_dim, num_layers):
        super().__init__(dictionary)
        self.dictionary = dictionary
        self.encoders = nn.ModuleList(encoders)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, src_tokens, src_lengths):
        # Fetch language IDs and remove them from src_tokens
        # Language IDs are on the right
        lang_ids = (
            src_tokens[:, -1] - pytorch_translate_data.MULTILING_DIALECT_ID_OFFSET
        )
        src_tokens = src_tokens[:, :-1]
        src_lengths -= 1
        # Create tensors for collecting encoder outputs
        bsz, seq_len = src_tokens.size()[:2]
        all_encoder_outs = torch.zeros(seq_len, bsz, self.hidden_dim).cuda()
        all_final_hidden = torch.zeros(self.num_layers, bsz, self.hidden_dim).cuda()
        all_final_cell = torch.zeros(self.num_layers, bsz, self.hidden_dim).cuda()
        # We cannot use zeros_like() for src_lengths because dtype changes
        # from LongInt to Int
        all_src_lengths = torch.zeros(bsz, dtype=torch.int).cuda()
        all_src_tokens = torch.zeros_like(src_tokens)
        for lang_id, encoder in enumerate(self.encoders):
            indices = torch.nonzero(lang_ids == lang_id)
            lang_bsz = indices.size(0)
            if lang_bsz == 0:  # Language not in this batch
                for p in encoder.parameters():
                    p.grad = torch.zeros_like(p.data)
                continue
            indices = indices.squeeze(1)
            (
                lang_encoder_outs,
                lang_final_hidden,
                lang_final_cell,
                lang_src_lengths,
                lang_src_tokens,
            ) = encoder(src_tokens[indices], src_lengths[indices])
            lang_seq_len = lang_encoder_outs.size(0)
            all_encoder_outs[:lang_seq_len, indices, :] = lang_encoder_outs
            all_final_hidden[:, indices, :] = lang_final_hidden
            all_final_cell[:, indices, :] = lang_final_cell
            all_src_lengths[indices] = lang_src_lengths
            all_src_tokens[indices] = lang_src_tokens
        return (
            all_encoder_outs,
            all_final_hidden,
            all_final_cell,
            all_src_lengths,
            all_src_tokens,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class MultilingualDecoder(FairseqIncrementalDecoder):
    """Multilingual decoder."""

    def __init__(self, dictionary, decoders, hidden_dim):
        super().__init__(dictionary)
        self.decoders = nn.ModuleList(decoders)
        self.hidden_dim = hidden_dim
        self.max_vocab_size = max(len(d.dictionary) for d in decoders)

    def forward(
        self,
        input_tokens,
        encoder_out,
        incremental_state=None,
        possible_translation_tokens=None,
    ):
        if input_tokens.size(1) <= 1:
            # This happens in the first time step of beam search. We return
            # flat scores, and wait for the real work in the next time step
            bsz = input_tokens.size(0)
            return (
                torch.zeros(bsz, 1, self.max_vocab_size).cuda(),
                torch.zeros(bsz, 1, encoder_out[0].size(0)).cuda(),
                None,
            )
        # Vocab reduction not implemented
        assert possible_translation_tokens is None
        # Fetch language IDs and remove them from input_tokens
        # Token sequences start with <GO-token> <lang-id> ...
        lang_ids = (
            input_tokens[:, 1] - pytorch_translate_data.MULTILING_DIALECT_ID_OFFSET
        )
        if input_tokens.size(1) > 2:
            input_tokens = torch.cat([input_tokens[:, :1], input_tokens[:, 2:]], dim=1)
        else:
            input_tokens = input_tokens[:, :1]

        bsz, seq_len = input_tokens.size()[:2]
        if incremental_state is None:
            incremental_state = {lang_id: None for lang_id in range(len(self.decoders))}
        else:
            seq_len = 1
        # Create tensors for collecting encoder outputs
        # +1 for language ID
        all_logits = torch.zeros(bsz, seq_len + 1, self.max_vocab_size).cuda()
        all_attn_scores = torch.zeros(bsz, seq_len, encoder_out[0].size(0)).cuda()
        for lang_id, decoder in enumerate(self.decoders):
            if lang_id not in incremental_state:
                incremental_state[lang_id] = {}
            indices = torch.nonzero(lang_ids == lang_id)
            lang_bsz = indices.size(0)
            if lang_bsz == 0:  # Language not in this batch
                for p in decoder.parameters():
                    p.grad = torch.zeros_like(p.data)
                continue
            indices = indices.squeeze(1)
            max_source_length = torch.max(encoder_out[3][indices])
            lang_encoder_out = (
                encoder_out[0][:max_source_length, indices, :],
                encoder_out[1][:, indices, :],
                encoder_out[2][:, indices, :],
                encoder_out[3][indices],
                encoder_out[4][indices, :max_source_length],
            )

            lang_logits, lang_attn_scores, _ = decoder(
                input_tokens[indices], lang_encoder_out, incremental_state[lang_id]
            )
            all_attn_scores[indices, :, :max_source_length] = lang_attn_scores
            all_logits[indices, 1:, : lang_logits.size(2)] = lang_logits
        incremental_state["lang_ids"] = lang_ids
        return all_logits, all_attn_scores, None

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        if not incremental_state:
            return
        bsz = new_order.size(0)
        for lang_id, decoder in enumerate(self.decoders):
            indices = torch.nonzero(incremental_state["lang_ids"] == lang_id)
            lang_bsz = indices.size(0)
            if lang_bsz > 0:
                if lang_bsz == bsz:
                    lang_new_order = new_order
                else:
                    lang_new_order = utils.densify(new_order[indices.squeeze(1)])
                decoder.reorder_incremental_state(
                    incremental_state[lang_id], lang_new_order
                )

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number
