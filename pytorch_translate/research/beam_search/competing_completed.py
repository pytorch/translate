#!/usr/bin/env python3

import math

import torch
from fairseq.models import FairseqIncrementalDecoder
from pytorch_translate.beam_decode import SequenceGenerator


class CompetingCompletedSequenceGenerator(SequenceGenerator):
    """Beam search which keeps completed hypotheses in the beam.

    This is an alternative beam search implementation which is more similar to
    some of the beam search implementations in the literature (cf. Nematus,
    Blocks, tensor2tensor, TF NMT tutorial...). This implementation keeps
    completed hypotheses as active hypotheses in the next time step. Thus, they
    will keep competing against longer hypotheses. Beam search terminates if all
    active hypotheses are completed (i.e. end with EOS).

    Experiments have shown that this implementation and the main implementation
    in `beam_search.py` produce almost the same translations. We're leaving this
    under research/ because experimentation did not show a noticeable difference
    in translation quality from the existing implementation.
    """

    def _generate(
        self,
        encoder_input,
        beam_size=None,
        maxlen=None,
        prefix_tokens=None,
        extra_info=True,
    ):
        """Run beam search.

        Args:
            encoder_input: 2-tuple (tokens, length) of int tensors containing
                the source tokens and the source sentence lengths.
            beam_size: beam size (if None, use self.beam_size)
            maxlen: Maximum target sentence length (if None, use self.maxlen)
            prefix_tokens: None or [bsz, prefix_length] int tensor with
                translation prefixes. All generated translations will be
                constrained to these prefixes.
            extra_info: If true, output additional information like alignment,
                attentions, and positional scores.

        Returns:
            A list of lists, containing the n-best translations for each
            batch entry. The translations are represented as dictionary with
            keys tokens, score, attention, alignment, positional_scores.
        """
        src_tokens = encoder_input[0]
        bsz, srclen = src_tokens.size()
        maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen
        beam_size = beam_size if beam_size is not None else self.beam_size

        # Encode
        encoder_outs, incremental_states = self._encode(encoder_input, beam_size)

        hypo_tokens = src_tokens.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
        hypo_tokens_buf = hypo_tokens.clone()
        hypo_tokens[:, 0] = self.eos
        cand_scores = src_tokens.new(bsz, beam_size).float().fill_(0)
        attn_list = []
        cand_scores_list = []
        cand_beams_list = []
        new_order = None
        # For example, new_order_offsets for bsz=3, beam_size=5 is
        # [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10]
        new_order_offsets = (
            torch.arange(0, bsz * beam_size, step=beam_size, dtype=torch.long)
            .view(bsz, 1)
            .repeat(1, beam_size)
            .view(-1)
            .cuda()
        )
        for step in range(maxlen + 1):  # one extra step for EOS marker
            num_finished = 0
            if step > 0:
                eos_indices = torch.nonzero(hypo_tokens[:, step] == self.eos)
                num_finished = eos_indices.size(0)
                if num_finished == bsz * beam_size:
                    break
            self.reorder_states(new_order, incremental_states)
            word_scores, avg_attn, possible_translation_tokens = self._decode(
                hypo_tokens[:, : step + 1], encoder_outs, incremental_states
            )
            attn_list.append(avg_attn.view(bsz, beam_size, -1))
            self.add_rewards(word_scores, step, possible_translation_tokens)
            if step == 0:  # Use only first beam in first time step
                word_scores.view(bsz, beam_size, -1)[:, 1:, :] = -math.inf
            elif num_finished > 0:  # make sure EOS is continued with EOS
                word_scores[eos_indices] = -math.inf
                word_scores[eos_indices, self.eos] = 0.0
            if prefix_tokens is not None and step < prefix_tokens.size(1):
                self.constrain_tokens(word_scores, prefix_tokens[step])
            if step >= maxlen:  # Force EOS on all hypos
                self.constrain_tokens(
                    word_scores, hypo_tokens.new_full((bsz * beam_size,), self.eos)
                )
            word_scores.add_(cand_scores.view(-1, 1))
            cand_scores, cand_indices, cand_beams = self.select_next_words(
                word_scores, bsz, beam_size, possible_translation_tokens
            )
            new_order = cand_beams.view(-1) + new_order_offsets
            torch.index_select(hypo_tokens, dim=0, index=new_order, out=hypo_tokens_buf)
            hypo_tokens_buf[:, step + 1] = cand_indices.view(-1)
            cand_scores_list.append(cand_scores)
            cand_beams_list.append(cand_beams)
            hypo_tokens, hypo_tokens_buf = hypo_tokens_buf, hypo_tokens
        build_hypo_fn = self.build_hypos if extra_info else self.build_hypos_fast
        return build_hypo_fn(
            hypo_tokens.view(bsz, beam_size, -1),
            cand_scores_list,
            attn_list,
            cand_beams_list,
        )

    def build_hypos(self, hypo_tokens, cand_scores_list, attn_list, cand_beams_list):
        bsz, beam_size, maxlen = hypo_tokens.size()
        seqlens = maxlen - torch.sum(hypo_tokens <= self.eos, dim=2)  # eos and pad
        all_hypos = []
        for batch_idx in range(bsz):
            hypos = []
            batch_cand_scores = self.backtrace(
                batch_idx, cand_beams_list, cand_scores_list
            )
            batch_attns = self.backtrace(batch_idx, cand_beams_list, attn_list)
            for i in range(beam_size):
                seqlen = seqlens[batch_idx, i]
                this_attns = torch.stack(batch_attns[i], dim=1)[:, : seqlen + 1]
                _, alignment = this_attns.max(dim=0)
                pos_scores = []
                prev = 0.0
                for pos in range(seqlen + 1):
                    pos_scores.append(batch_cand_scores[i][pos] - prev)
                    prev = batch_cand_scores[i][pos]
                hypos.append(
                    {
                        "tokens": hypo_tokens[batch_idx, i, 1 : seqlen + 2],
                        "score": cand_scores_list[-1][batch_idx, i],
                        "attention": this_attns,
                        "alignment": alignment,
                        "positional_scores": pos_scores,
                    }
                )
            all_hypos.append(hypos)
        return all_hypos

    def build_hypos_fast(
        self, hypo_tokens, cand_scores_list, attn_list, cand_beams_list
    ):
        bsz, beam_size, maxlen = hypo_tokens.size()
        seqlens = maxlen - torch.sum(hypo_tokens <= self.eos, dim=2)  # eos and pad
        all_hypos = []
        dummy_alignment = torch.LongTensor([1, 2, 3])
        for batch_idx in range(bsz):
            hypos = []
            for i in range(beam_size):
                seqlen = seqlens[batch_idx, i]
                hypos.append(
                    {
                        "tokens": hypo_tokens[batch_idx, i, 1 : seqlen + 2],
                        "score": cand_scores_list[-1][batch_idx, i],
                        "attention": None,
                        "alignment": dummy_alignment,
                        "positional_scores": None,
                    }
                )
            all_hypos.append(hypos)
        return all_hypos

    def backtrace(self, batch_idx, backpointers_list, elements_list):
        beam_size = backpointers_list[0].size(1)
        backtraced = [[elements_list[-1][batch_idx, i]] for i in range(beam_size)]
        hypo_ptrs = list(range(beam_size))
        pos = -1
        for backpointers in reversed(backpointers_list[1:]):
            pos -= 1
            for i in range(beam_size):
                hypo_ptrs[i] = backpointers[batch_idx][hypo_ptrs[i]]
                backtraced[i].append(elements_list[pos][batch_idx, hypo_ptrs[i]])
        for l in backtraced:
            l.reverse()
        return backtraced

    def select_next_words(
        self, word_scores, bsz, beam_size, possible_translation_tokens
    ):
        cand_scores, cand_indices = torch.topk(word_scores.view(bsz, -1), k=beam_size)
        possible_tokens_size = self.vocab_size
        if possible_translation_tokens is not None:
            possible_tokens_size = possible_translation_tokens.size(0)
        cand_beams = torch.div(cand_indices, possible_tokens_size)
        cand_indices.fmod_(possible_tokens_size)
        # Handle vocab reduction
        if possible_translation_tokens is not None:
            possible_translation_tokens = possible_translation_tokens.view(
                1, possible_tokens_size
            ).expand(cand_indices.size(0), possible_tokens_size)
            cand_indices = torch.gather(
                possible_translation_tokens, dim=1, index=cand_indices, out=cand_indices
            )
        return cand_scores, cand_indices, cand_beams

    def constrain_tokens(self, word_scores, permitted_tokens):
        """Modifies word_scores such that hypos are expanded with the tokens
        in `permitted_tokens`.

        Args:
            word_scores: [bsz*beam_size, vocab_size] float tensor with
                cumulative word scores (logprobs plus hypo scores plus rewards)
            permitted_tokens: [bsz*beam_size] int tensor of tokens. We set all
                entries in `word_scores` to -inf except the permitted tokens.
        """
        permitted_tokens = permitted_tokens.unsqueeze(1)
        permitted_scores = torch.gather(word_scores, dim=1, index=permitted_tokens)
        word_scores[:, :] = -math.inf
        word_scores.scatter_(dim=1, index=permitted_tokens, src=permitted_scores)

    def reorder_states(self, new_order, incremental_states):
        if new_order is None:
            return
        for model in self.models:
            if isinstance(model.decoder, FairseqIncrementalDecoder):
                model.decoder.reorder_incremental_state(
                    incremental_states[model], new_order
                )

    def add_rewards(self, word_scores, step, possible_translation_tokens):
        """Compute the accumulated scores for each word.

        Args:
            word_scores: [bsz*beam_size, vocab_size] float tensor with cumulative
                word scores (logprobs plus hypo scores)
            step (int): time step
            possible_translation_tokens: For vocab reduction
        """
        word_scores[:, self.pad] = -math.inf  # never select pad

        # apply unk reward
        if possible_translation_tokens is None:
            unk_index = self.unk
        else:
            unk_index = torch.nonzero(possible_translation_tokens == self.unk)[0, 0]
        word_scores[:, unk_index] += self.unk_reward

        # external lexicon reward
        word_scores[:, self.lexicon_indices] += self.lexicon_reward

        word_scores += self.word_reward
        if step >= self.minlen:
            word_scores[:, self.eos] -= self.word_reward
        else:
            word_scores[:, self.eos] = -math.inf
