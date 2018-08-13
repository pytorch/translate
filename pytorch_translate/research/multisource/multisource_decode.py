#!/usr/bin/env python3

import math

import torch
from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder


class MultiSourceSequenceGenerator(torch.nn.Module):
    # We use the 1st source sentence as reference alignment
    align_to = 0

    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        minlen=1,
        maxlen=None,
        stop_early=True,
        normalize_scores=True,
        len_penalty=0,
        unk_reward=0,
        lexicon_reward=0,
        retain_dropout=False,
        word_reward=0,
        model_weights=None,
        use_char_source=False,
        align_to=1,
    ):
        """Generates translations from multiple source sentences

        This only supports one model for now.

        Args:
            models: List of FairseqModel objects. Each one must implement
                expand_encoder_output() method to replicate encoder outputs.
                For now only one model is supported
            min/maxlen: The length of the generated output will be bounded by
                minlen and maxlen (not including the end-of-sentence marker).
            stop_early: Stop generation immediately after we finalize beam_size
                hypotheses, even though longer hypotheses might have better
                normalized scores.
            normalize_scores: Normalize scores by the length of the output.
            word_reward: add this value to score each token except EOS
                (an alternative method to len_penalty for encouraging longer
                output)
            model_weights: None or list of Python floats of the same length as
                `models` with ensemble interpolation weights.
            use_char_source: if True, encoder inputs consist of (src_tokens,
                src_lengths, char_inds, word_lengths)
        """
        self.models = models
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        self.minlen = minlen
        max_decoder_len = min(m.max_decoder_positions() for m in self.models)
        self.maxlen = (
            max_decoder_len if maxlen is None else min(maxlen, max_decoder_len)
        )
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_reward = unk_reward
        self.lexicon_reward = lexicon_reward
        self.lexicon_indices = tgt_dict.lexicon_indices_list()
        self.retain_dropout = retain_dropout
        self.word_reward = word_reward
        if model_weights is not None:
            assert len(models) == len(model_weights)
            self.model_weights = model_weights
        else:
            self.model_weights = [1.0 / len(models)] * len(models)
        self.use_char_source = use_char_source

    def cuda(self):
        for model in self.models:
            model.cuda()
        return self

    def generate_batched_itr(
        self,
        data_itr,
        beam_size=None,
        maxlen_a=0.0,
        maxlen_b=None,
        cuda=False,
        timer=None,
        prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual translations.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """
        if maxlen_b is None:
            maxlen_b = self.maxlen

        for sample in data_itr:
            if cuda:
                s = utils.move_to_cuda(sample)
            input = s["net_input"]
            # Take the max source length to compute the max target length
            srclen = input["src_tokens"].size(1)
            # FIXME: handle characters properly
            if self.use_char_source:
                raise ValueError(
                    "Character level encoder is not supported yet for "
                    "multisource sentences."
                )
            encoder_inputs = (input["src_tokens"], input["src_lengths"])
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    encoder_inputs,
                    srcs_ids=input["src_ids"],
                    beam_size=beam_size,
                    maxlen=int(maxlen_a * srclen + maxlen_b),
                    prefix_tokens=s["target"][:, :prefix_size]
                    if prefix_size > 0
                    else None,
                )
            if timer is not None:
                timer.stop(s["ntokens"])
            for i, id in enumerate(s["id"]):
                src = input["src_tokens"].index_select(
                    0, input["src_ids"][self.align_to]
                )
                # remove padding from ref
                ref = utils.strip_pad(s["target"][i, :], self.pad)
                yield id, src, ref, hypos[i]

    def generate(
        self,
        encoder_inputs,
        srcs_ids,
        beam_size=None,
        maxlen=None,
        prefix_tokens=None,
        src_weights=None,
    ):
        """Generate a batch of translations."""
        with torch.no_grad():
            return self._generate(
                encoder_inputs, srcs_ids, beam_size, maxlen, prefix_tokens, src_weights
            )

    def _generate(
        self,
        encoder_inputs,
        srcs_ids,
        beam_size=None,
        maxlen=None,
        prefix_tokens=None,
        src_weights=None,
    ):
        """Generates a translation from multiple source sentences"""
        n_srcs = len(srcs_ids)
        srcs_tokens = encoder_inputs[0]
        align_src_tokens = srcs_tokens.index_select(0, srcs_ids[self.align_to])

        bsz, srclen = align_src_tokens.size()
        maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen

        # the max beam size is the dictionary size - 1, since we never select pad
        beam_size = beam_size if beam_size is not None else self.beam_size
        assert (
            beam_size < self.vocab_size
        ), "Beam size must be smaller than target vocabulary"

        # Encode
        encoder_outs = self._encode(encoder_inputs, beam_size, srcs_ids)
        incremental_states = self._init_incremental_states(n_srcs)

        # initialize buffers
        scores = align_src_tokens.new(bsz * beam_size, maxlen + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = align_src_tokens.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos

        # may differ from input length
        src_encoding_len = encoder_outs[self.align_to][0][0].size(0)

        attn = scores.new(bsz * beam_size, src_encoding_len, maxlen + 2)
        attn_buf = attn.clone()

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{"idx": None, "score": -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == maxlen or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= (maxlen + 1) ** self.len_penalty
                if worst_finalized[sent]["score"] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[
                :, 1 : step + 2
            ]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            sents_seen = set()
            for i, (idx, score) in enumerate(
                zip(bbsz_idx.tolist(), eos_scores.tolist())
            ):
                sent = idx // beam_size
                sents_seen.add(sent)

                def get_hypo():
                    _, alignment = attn_clone[i].max(dim=0)
                    return {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "attention": attn_clone[i],  # src_len x tgt_len
                        "alignment": alignment,
                        "positional_scores": pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]["score"]:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]["idx"]
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(
                        enumerate(finalized[sent]), key=lambda r: r[1]["score"]
                    )
                    worst_finalized[sent] = {"score": s["score"], "idx": idx}

            # return number of hypotheses finished this step
            num_finished = 0
            for sent in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                    finished[sent] = True
                    num_finished += 1
            return num_finished

        reorder_state = None
        for step in range(maxlen + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                for model_id, model in enumerate(self.models):
                    if isinstance(model.decoder, FairseqIncrementalDecoder):
                        for src_id in range(n_srcs):
                            model.decoder.reorder_incremental_state(
                                incremental_states[(src_id, model_id)], reorder_state
                            )
            # Run decoder for one step
            logprobs, avg_attn, possible_translation_tokens = self._decode(
                tokens[:, : step + 1], encoder_outs, incremental_states, n_srcs
            )

            if step == 0:
                # at the first step all hypotheses are equally likely, so use
                # only the first beam
                logprobs = logprobs.unfold(0, 1, beam_size).squeeze(2).contiguous()
                scores = scores.type_as(logprobs)
                scores_buf = scores_buf.type_as(logprobs)
            else:
                # make probs contain cumulative scores for each hypothesis
                logprobs.add_(scores[:, step - 1].view(-1, 1))
            logprobs[:, self.pad] = -math.inf  # never select pad

            # apply unk reward
            if possible_translation_tokens is None:
                unk_index = self.unk
            else:
                unk_index = torch.nonzero(possible_translation_tokens == self.unk)[0, 0]
            logprobs[:, unk_index] += self.unk_reward

            # external lexicon reward
            logprobs[:, self.lexicon_indices] += self.lexicon_reward

            logprobs += self.word_reward
            logprobs[:, self.eos] -= self.word_reward

            # Record attention scores
            attn[:, :, step + 1].copy_(avg_attn)

            cand_scores = buffer("cand_scores", type_of=scores)
            cand_indices = buffer("cand_indices")
            cand_beams = buffer("cand_beams")
            eos_bbsz_idx = buffer("eos_bbsz_idx")
            eos_scores = buffer("eos_scores", type_of=scores)
            if step < maxlen:
                if prefix_tokens is not None and step < prefix_tokens.size(1):
                    logprobs_slice = logprobs.view(bsz, -1, logprobs.size(-1))[:, 0, :]
                    cand_scores = torch.gather(
                        logprobs_slice, dim=1, index=prefix_tokens[:, step].view(-1, 1)
                    ).expand(-1, cand_size)
                    cand_indices = (
                        prefix_tokens[:, step].view(-1, 1).expand(bsz, cand_size)
                    )
                    cand_beams.resize_as_(cand_indices).fill_(0)
                else:
                    # take the best 2 x beam_size predictions. We'll choose the first
                    # beam_size of these which don't predict eos to continue with.
                    torch.topk(
                        logprobs.view(bsz, -1),
                        k=min(
                            cand_size, logprobs.view(bsz, -1).size(1) - 1
                        ),  # -1 so we never select pad
                        out=(cand_scores, cand_indices),
                    )

                    possible_tokens_size = self.vocab_size
                    if possible_translation_tokens is not None:
                        possible_tokens_size = possible_translation_tokens.size(0)
                    # cand_indices has values in [0, vocab_size * beam_size]
                    # the following does euclidean division bu vocab_size
                    # to retrieve the beam and word id of each candidate
                    torch.div(cand_indices, possible_tokens_size, out=cand_beams)
                    cand_indices.fmod_(possible_tokens_size)
                    # Handle vocab reduction
                    if possible_translation_tokens is not None:
                        possible_translation_tokens = possible_translation_tokens.view(
                            1, possible_tokens_size
                        ).expand(cand_indices.size(0), possible_tokens_size)
                        cand_indices = torch.gather(
                            possible_translation_tokens,
                            dim=1,
                            index=cand_indices,
                            out=cand_indices,
                        )
            else:
                # finalize all active hypotheses once we hit maxlen
                # pick the hypothesis with the highest log prob of EOS right now
                torch.sort(
                    logprobs[:, self.eos],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= finalize_hypos(step, eos_bbsz_idx, eos_scores)
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add_(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)
            if step >= self.minlen:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,
                )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_scores,
                    )
                    num_remaining_sent -= finalize_hypos(
                        step, eos_bbsz_idx, eos_scores, cand_scores
                    )

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < maxlen

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer("active_mask")
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer("active_hypos"), buffer("_ignore")
            torch.topk(
                active_mask,
                k=beam_size,
                dim=1,
                largest=False,
                out=(_ignore, active_hypos),
            )
            active_bbsz_idx = buffer("active_bbsz_idx")
            torch.gather(cand_bbsz_idx, dim=1, index=active_hypos, out=active_bbsz_idx)
            active_scores = torch.gather(
                cand_scores,
                dim=1,
                index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )
            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, : step + 1],
                dim=0,
                index=active_bbsz_idx,
                out=tokens_buf[:, : step + 1],
            )
            torch.gather(
                cand_indices,
                dim=1,
                index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step],
                    dim=0,
                    index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores,
                dim=1,
                index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            torch.index_select(
                attn[:, :, : step + 2],
                dim=0,
                index=active_bbsz_idx,
                out=attn_buf[:, :, : step + 2],
            )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(bsz):
            finalized[sent] = sorted(
                finalized[sent], key=lambda r: r["score"], reverse=True
            )

        return finalized

    def _init_incremental_states(self, n_srcs):
        incremental_states = {}
        for src_id in range(n_srcs):
            for model_id, model in enumerate(self.models):
                if isinstance(model.decoder, FairseqIncrementalDecoder):
                    incremental_states[(src_id, model_id)] = {}
                else:
                    incremental_states[(src_id, model_id)] = None
        return incremental_states

    def _encode(self, encoder_inputs, beam_size, srcs_ids):
        encoder_outs = [[] for _ in range(len(srcs_ids))]

        def pick_src_encodings(encoder_out, src_ids):
            (
                unpacked_output,
                final_hiddens,
                final_cells,
                src_lengths,
                src_tokens,
            ) = encoder_out
            unpacked_output = unpacked_output.index_select(1, src_ids)
            final_hiddens = final_hiddens.index_select(1, src_ids)
            final_cells = final_cells.index_select(1, src_ids)
            src_lengths = src_lengths.index_select(0, src_ids)
            src_tokens = src_tokens.index_select(0, src_ids)
            max_src_len = src_lengths.data.max()
            return (
                unpacked_output[:max_src_len, :, :],
                final_hiddens,
                final_cells,
                src_lengths,
                src_tokens[:, :max_src_len],
            )

        # Enumerate all input/model pair
        # At this point encoder_input is a tuple of srcs_tokens and srcs_lengths
        # So we call zip(*) on it to get a list of token, length tuples
        for model in self.models:
            if not self.retain_dropout:
                model.eval()
            encoder_out = model.encoder(*encoder_inputs)
            for k, src_ids in enumerate(srcs_ids):
                # Get the encodings corresponding to each source sentence
                encoder_out_k = pick_src_encodings(encoder_out, src_ids)
                # expand outputs for each example beam_size times
                encoder_out_k = model.expand_encoder_output(encoder_out_k, beam_size)
                encoder_outs[k].append(encoder_out_k)
        return encoder_outs

    def _decode(self, tokens, encoder_outs, incremental_states, n_srcs=1):
        # Source sentences are weighted equally (for now)
        srcs_weights = [1 / n_srcs] * n_srcs

        avg_probs = None
        avg_attn = None
        for src_id, src_weight in enumerate(srcs_weights):
            for model_id, (model_weight, model) in enumerate(
                zip(self.model_weights, self.models)
            ):
                with torch.no_grad():
                    encoder_out = encoder_outs[src_id][model_id]
                    incremental_state = incremental_states[(src_id, model_id)]
                    decoder_out = list(
                        model.decoder(tokens, encoder_out, incremental_state)
                    )
                    decoder_out[0] = decoder_out[0][:, -1, :]
                    attn = decoder_out[1]
                    if len(decoder_out) == 3:
                        possible_translation_tokens = decoder_out[2]
                    else:
                        possible_translation_tokens = None
                probs = (
                    src_weight
                    * model_weight
                    * model.get_normalized_probs(decoder_out, log_probs=False)
                )
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs.add_(probs)
                if attn is not None and src_id == self.align_to:
                    attn = attn[:, -1, :]
                    if avg_attn is None:
                        avg_attn = attn
                    else:
                        avg_attn.add_(attn)
        avg_probs.log_()
        if avg_attn is not None:
            avg_attn.div_(len(self.models))

        return avg_probs, avg_attn, possible_translation_tokens
