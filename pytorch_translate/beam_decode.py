#!/usr/bin/env python3

import math
from typing import List, Tuple

import torch
from fairseq import search, utils
from fairseq.models import FairseqIncrementalDecoder
from pytorch_translate import utils as pytorch_translate_utils
from torch import Tensor


class SequenceGenerator(object):
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
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
        diversity_sibling_gamma=0.0,
        sampling=False,
        sampling_topk=-1,
        temperature=1,
    ):
        """Generates translations of a given source sentence.
        Args:
            models: List of FairseqEncoderDecoderModel objects. Each one must
                implement reorder_encoder_output() method to replicate encoder
                outputs.
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
            diverse_beam_groups: number of groups for Diverse Beam Search
                (-1 by default is vanilla beam search)
            diverse_beam_strength: strength of diversity penalty for Diverse
                Beam Search.
            diversity_sibling_gamma: The diversity rate of sibling rank (-0.0 by default
               to disable sibling rank penalty)
            sampling (bool, optional): sample outputs instead of beam search
                (default: False)
            sampling_topk (int, optional): only sample among the top-k choices
                at each step (default: -1)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
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
        self.temperature = temperature
        self.word_reward = word_reward
        if model_weights is not None:
            assert len(models) == len(model_weights)
            self.model_weights = model_weights
        else:
            self.model_weights = [1.0 / len(models)] * len(models)
        self.use_char_source = use_char_source
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert temperature > 0, "--temperature must be greater than 0"
        if sampling:
            self.search = search.Sampling(tgt_dict, sampling_topk)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(
                tgt_dict, diverse_beam_groups, diverse_beam_strength
            )
        else:
            self.search = search.BeamSearch(tgt_dict)
        self.diversity_sibling_gamma = diversity_sibling_gamma

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
            if "net_input" not in sample:
                continue

            if cuda:
                s = utils.move_to_cuda(sample)
            else:
                s = sample
            input = s["net_input"]
            srclen = input["src_tokens"].size(1)
            if self.use_char_source:
                encoder_input = {
                    k: v
                    for k, v in input.items()
                    if k in ["src_tokens", "src_lengths", "char_inds", "word_lengths"]
                }
            else:
                encoder_input = {
                    k: v for k, v in input.items() if k in ["src_tokens", "src_lengths"]
                }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    encoder_input=encoder_input,
                    beam_size=beam_size,
                    maxlen=int(maxlen_a * srclen + maxlen_b),
                    prefix_tokens=s["target"][:, :prefix_size]
                    if prefix_size > 0
                    else None,
                )
            if timer is not None:
                timer.stop(s["ntokens"])
            for i, id in enumerate(s["id"]):
                # remove padding
                src = utils.strip_pad(input["src_tokens"][i, :], self.pad)
                ref = utils.strip_pad(s["target"][i, :], self.pad)
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, encoder_input, beam_size=None, maxlen=None, prefix_tokens=None):
        encoder_inputs = self.prepare_encoder_inputs(encoder_input)
        encoder_outs, incremental_states = self._encode(encoder_input=encoder_inputs)
        return self._decode_target(
            encoder_input,
            encoder_outs,
            incremental_states,
            self.diversity_sibling_gamma,
            beam_size,
            maxlen,
            prefix_tokens,
        )

    def prepare_encoder_inputs(self, encoder_input):
        if self.use_char_source:
            encoder_inputs = (
                encoder_input["src_tokens"],
                encoder_input["src_lengths"],
                encoder_input["char_inds"],
                encoder_input["word_lengths"],
            )
        else:
            encoder_inputs = (encoder_input["src_tokens"], encoder_input["src_lengths"])
        return encoder_inputs

    def _build_constraints(self, src_tokens, beam_size):
        """
        Stub functions for adding application specific constraint checks on
        the candidates being generated during beam search. This and the below
        stub functions can be implemented in a child class without needing to
        touch the actual beam search code
        """
        pass

    def _apply_constraint_penalty(self, scores):
        pass

    def _update_constraints(self, constraints, next_tokens, idx):
        pass

    def _reorder_constraints(self, constraints, new_indices):
        pass

    def _apply_eos_constraints(self, constraints, eos_bbsz_idx, eos_scores):
        pass

    def _finalize_constrained_results(self, finalized, device):
        pass

    def _decode_target(
        self,
        encoder_input,
        encoder_outs,
        incremental_states,
        diversity_sibling_gamma=0.0,
        beam_size=None,
        maxlen=None,
        prefix_tokens=None,
    ):
        src_tokens_tensor = pytorch_translate_utils.get_source_tokens_tensor(
            encoder_input["src_tokens"]
        )
        beam_size = beam_size if beam_size is not None else self.beam_size
        bsz = src_tokens_tensor.size(0)
        reorder_indices = (
            torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1).long()
        )
        for i, model in enumerate(self.models):
            encoder_outs[i] = model.encoder.reorder_encoder_out(
                encoder_out=encoder_outs[i],
                new_order=reorder_indices.type_as(src_tokens_tensor),
            )
        maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen
        # initialize buffers
        scores = src_tokens_tensor.new(bsz * beam_size, maxlen + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens_tensor.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos

        # may differ from input length
        if isinstance(encoder_outs[0], (list, tuple)):
            src_encoding_len = encoder_outs[0][0].size(0)
        elif isinstance(encoder_outs[0], dict):
            if isinstance(encoder_outs[0]["encoder_out"], tuple):
                # Fairseq compatibility
                src_encoding_len = encoder_outs[0]["encoder_out"][0].size(1)
            else:
                src_encoding_len = encoder_outs[0]["encoder_out"].size(0)

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

        # init constraints
        constraints = self._build_constraints(src_tokens_tensor, beam_size)

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
        possible_translation_tokens = None
        for step in range(maxlen + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                for model in self.models:
                    if isinstance(model.decoder, FairseqIncrementalDecoder):
                        model.decoder.reorder_incremental_state(
                            incremental_states[model], reorder_state
                        )
            # Run decoder for one step
            logprobs, avg_attn, possible_translation_tokens = self._decode(
                tokens[:, : step + 1],
                encoder_outs,
                incremental_states,
                possible_translation_tokens,
            )

            logprobs[:, self.pad] = -math.inf  # never select pad
            # apply unk reward
            if possible_translation_tokens is None:
                # No vocab reduction, so unk is represented by self.unk at
                # position self.unk
                unk_index = self.unk
                logprobs[:, unk_index] += self.unk_reward
            else:
                # When we use vocab reduction, the token value self.unk may not
                # be at the position self.unk, but somewhere else in the list
                # of possible_translation_tokens. It's also possible not to
                # show up in possible_translation_tokens at all, meaning we
                # can't generate an unk.
                unk_pos = torch.nonzero(possible_translation_tokens == self.unk)
                if unk_pos.size()[0] != 0:
                    # only add unk_reward if unk index appears in
                    # possible_translation_tokens
                    unk_index = unk_pos[0][0]
                    logprobs[:, unk_index] += self.unk_reward
            # external lexicon reward
            logprobs[:, self.lexicon_indices] += self.lexicon_reward

            logprobs += self.word_reward
            logprobs[:, self.eos] -= self.word_reward
            # Record attention scores
            if avg_attn is not None:
                attn[:, :, step + 1].copy_(avg_attn)

            cand_scores = buffer("cand_scores", type_of=scores)
            cand_indices = buffer("cand_indices")
            cand_beams = buffer("cand_beams")
            eos_bbsz_idx = buffer("eos_bbsz_idx")
            eos_scores = buffer("eos_scores", type_of=scores)
            scores = scores.type_as(logprobs)
            scores_buf = scores_buf.type_as(logprobs)

            if step < maxlen:
                self._apply_constraint_penalty(scores)  # stub call
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
                    possible_tokens_size = self.vocab_size
                    if possible_translation_tokens is not None:
                        possible_tokens_size = possible_translation_tokens.size(0)
                    if diversity_sibling_gamma > 0:
                        logprobs = self.diversity_sibling_rank(
                            logprobs.view(bsz, -1, possible_tokens_size),
                            diversity_sibling_gamma,
                        )
                    cand_scores, cand_indices, cand_beams = self.search.step(
                        step,
                        logprobs.view(bsz, -1, possible_tokens_size),
                        scores.view(bsz, beam_size, -1)[:, :, :step],
                    )
                    # vocabulary reduction
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
                logprobs.add_(scores[:, step - 1].view(-1, 1))
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
                    self._apply_eos_constraints(constraints, eos_bbsz_idx, eos_scores)
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
            # update constraints for next step
            constraints = self._reorder_constraints(constraints, active_bbsz_idx)
            self._update_constraints(constraints, tokens_buf[:, step + 1], step)
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
        self._finalize_constrained_results(finalized, scores.device)
        return finalized

    def _encode(self, encoder_input):
        encoder_outs = []
        incremental_states = {}
        for model in self.models:
            if not self.retain_dropout:
                model.eval()
            if isinstance(model.decoder, FairseqIncrementalDecoder):
                incremental_states[model] = {}
            else:
                incremental_states[model] = None
            encoder_out = model.encoder(*encoder_input)
            encoder_outs.append(encoder_out)
        return encoder_outs, incremental_states

    @staticmethod
    def gather_probs(all_translation_tokens, all_probs):
        """
        Maps probabilities for multiple models with different output softmax
        dimensions to the same combined token space. This is a simplified
        example, normally probs would be in log space and would be size
        [bsz, len(possible_translation_tokens)]

        Model 1:
        possible_translation_tokens: [3, 7, 8, 9]
        probs: [0.25, 0.25, 0.25, 0.25]

        Model 2:
        possible_translation_tokens: [0, 3, 5]
        probs: [0.4, 0.5, 0.1]

        all_translation_tokens: [[3, 7, 8, 9], [0, 3, 5]]
        all_probs: [[0.25, 0.25, 0.25, 0.25], [0.4, 0.5, 0.1]]
        possible_translation_tokens = [0, 3, 5, 7, 8, 9] (order varies)
        mapped_probs for model 1: [0  , 0.25, 0  , 0.25, 0.25, 0.25]
        mapped_probs for model 2: [0.4, 0.5 , 0.1, 0   , 0   , 0]

        avg_probs = [0.4, 0.75, 0.1, 0.25, 0.25, 0.25] (order varies but
        corresponds to possible_translation_tokens)

        Inputs:
            all_translation_tokens: List[Optional[possible_translation_tokens]]
                where possible_translation_tokens is a flat Tensor representing
                the possible translation tokens from model output. Note that the
                possible_translation_tokens will be None only if vocab reduction
                was not used.
            all_probs: List[probs] where probs is a flat Tensor of normalized
                probs for each model output. If vocab reduction was not used,
                each probs list will be of length vocab size. Otherwise, each
                probs will be the same length as that model's
                possible_translation_tokens

        Returns:
            avg_probs: average probabilities of tokens from a merged list of
                possible_translation_tokens from every model.
            possible_translation_tokens: merged list of
                possible_translation_tokens from every model.
        """

        assert len(all_translation_tokens) == len(all_probs), (
            f"Number of possible_translation_tokens tensors in "
            f"all_translation_tokens list -- got length "
            f"{len(all_translation_tokens)} -- should match the number of "
            f"probs tensors in all_probs list -- got length {len(all_probs)}.\n"
            f"all_translation_tokens: {all_translation_tokens}\n"
            f"all_probs: {all_probs}"
        )
        possible_translation_tokens = None
        inv_indices_per_model = [None] * len(all_translation_tokens)
        if all_translation_tokens[0] is not None:
            # Get unique translation tokens out of all the
            # possible_translation_tokens for every model.
            # inverse indices for the example above: [5, 4, 2, 1, 3, 5, 0]
            possible_translation_tokens, inverse_indices = torch.unique(
                torch.cat(all_translation_tokens, dim=0),
                sorted=False,
                return_inverse=True,
            )
            # softmax_sizes for the example above: [4, 3]
            softmax_sizes = [
                translation_tokens.size(0)
                for translation_tokens in all_translation_tokens
            ]
            inv_indices_per_model = torch.split(
                inverse_indices, split_size_or_sections=softmax_sizes
            )

        avg_probs = None
        for inv_ind, probs in zip(inv_indices_per_model, all_probs):
            mapped_probs = probs
            if possible_translation_tokens is not None:
                # The corresponding model did not use vocab reduction if
                # possible_translation_tokens is None.
                mapped_probs = torch.zeros(
                    (probs.size(0), possible_translation_tokens.size(0)),
                    device=probs.device,
                )

                mapped_probs[:, inv_ind] = probs
            if avg_probs is None:
                avg_probs = mapped_probs
            else:
                avg_probs.add_(mapped_probs)
        return avg_probs, possible_translation_tokens

    def _decode(
        self, tokens, encoder_outs, incremental_states, possible_translation_tokens=None
    ):
        avg_attn = None
        all_translation_tokens = []
        all_log_probs = []
        for model_weight, model, encoder_out in zip(
            self.model_weights, self.models, encoder_outs
        ):
            with torch.no_grad():
                if (
                    possible_translation_tokens is not None
                    and len(possible_translation_tokens.shape) > 1
                ):
                    # reverse beam replication
                    possible_translation_tokens = possible_translation_tokens[0]

                decoder_out = list(
                    model.decoder(
                        tokens,
                        encoder_out,
                        incremental_states[model],
                        possible_translation_tokens=possible_translation_tokens,
                    )
                )
                decoder_out[0] = decoder_out[0][:, -1, :]
                if self.temperature != 1.0:
                    decoder_out[0].div_(self.temperature)
                attn = decoder_out[1]
                if len(decoder_out) == 3:
                    possible_translation_tokens = decoder_out[2]
                else:
                    possible_translation_tokens = None
                if (
                    hasattr(model.decoder, "adaptive_softmax")
                    and model.decoder.adaptive_softmax is not None
                ):
                    decoder_out[0] = decoder_out[0].unsqueeze(1)
                    # to use get_normalized_probs in adaptive softmax decoder
                    # the sample object is needed. During inference, the target
                    # should be set to None
                    log_probs = model.get_normalized_probs(
                        decoder_out, log_probs=True, sample={"target": None}
                    )
                    log_probs = model_weight * log_probs[:, -1, :]
                else:
                    log_probs = model.get_normalized_probs(decoder_out, log_probs=True)
                    log_probs = model_weight * log_probs
                all_translation_tokens.append(possible_translation_tokens)
                all_log_probs.append(log_probs)

            if attn is not None:
                attn = attn[:, -1, :].data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_log_probs, possible_translation_tokens = SequenceGenerator.gather_probs(
            all_translation_tokens=all_translation_tokens, all_probs=all_log_probs
        )
        if avg_attn is not None:
            avg_attn.div_(len(self.models))

        return avg_log_probs, avg_attn, possible_translation_tokens

    def diversity_sibling_rank(self, logprobs, gamma):
        """
        See "A Simple, Fast Diverse Decoding Algorithm for Neural Generation"
        for details
        """
        _, beam_size, vocab_size = logprobs.size()
        logprobs = logprobs.view(-1, vocab_size)
        # Keep consistent with beamsearch class in fairseq
        k = min(2 * beam_size, vocab_size)
        _, indices = torch.topk(logprobs, k)
        # Set diverse penalty as k for all words
        diverse_penalty = torch.ones_like(logprobs) * k
        diversity_sibling_rank = (
            torch.arange(0, k).view(-1, 1).expand(k, logprobs.size(0)).type_as(logprobs)
        )
        # Set diversity penalty accordingly for top-k words
        diverse_penalty[
            torch.arange(0, logprobs.size(0)).long(), indices.transpose(0, 1)
        ] = diversity_sibling_rank
        logprobs -= gamma * diverse_penalty
        return logprobs


class BeamDecode(torch.jit.ScriptModule):
    """
    Decodes the output of Beam Search to get the top hypotheses
    """

    def __init__(self, eos_token_id, length_penalty, nbest, beam_size, stop_at_eos):
        super().__init__()
        self.eos_token_id = torch.jit.Attribute(eos_token_id, int)
        self.length_penalty = torch.jit.Attribute(length_penalty, float)
        self.nbest = torch.jit.Attribute(nbest, int)
        self.beam_size = torch.jit.Attribute(beam_size, int)
        self.stop_at_eos = torch.jit.Attribute(int(stop_at_eos), int)

    @torch.jit.script_method
    @torch.no_grad()
    def forward(
        self,
        beam_tokens: Tensor,
        beam_scores: Tensor,
        token_weights: Tensor,
        beam_prev_indices: Tensor,
        num_steps: int,
    ) -> List[Tuple[Tensor, float, List[float], Tensor, Tensor]]:

        self._check_dimensions(
            beam_tokens, beam_scores, token_weights, beam_prev_indices, num_steps
        )

        end_states = self._get_all_end_states(
            beam_tokens, beam_scores, beam_prev_indices, num_steps
        )

        # outputs is list of the following for each hypothesis:
        # Tuple[Hypothesis, Hypothesis score, Token level scores, Attention Weights, Best indices]
        outputs = torch.jit.annotate(
            List[Tuple[Tensor, float, List[float], Tensor, Tensor]], []
        )

        for state_idx in range(len(end_states)):
            state = end_states[state_idx]
            hypothesis_score = float(state[0])
            beam_indices = self._get_output_steps_to_beam_indices(
                state, beam_prev_indices
            )
            beam_output = torch.jit.annotate(List[Tensor], [])
            token_level_scores = torch.jit.annotate(List[float], [])
            position = int(state[1])
            hyp_index = int(state[2])

            # best_indices represents the ending position of one hypothesis,
            # the first index corresponds num_step, the second corresponds beam_index
            best_indices = torch.tensor([position, hyp_index])
            back_alignment_weights = []

            assert position + 1 == len(beam_indices)
            pos = 1
            prev_beam_index = -1
            while pos < len(beam_indices):
                beam_index = beam_indices[pos]
                beam_output.append(beam_tokens[pos][beam_index])
                if pos == 1:
                    # beam_scores[0][:] are all 0s
                    token_level_scores.append(float(beam_scores[pos][beam_index]))
                else:
                    token_level_scores.append(
                        float(beam_scores[pos][beam_index])
                        - float(beam_scores[pos - 1][prev_beam_index])
                    )
                back_alignment_weights.append(token_weights[pos][beam_index].detach())
                prev_beam_index = beam_index
                pos += 1
            outputs.append(
                (
                    torch.stack(beam_output),
                    hypothesis_score,
                    token_level_scores,
                    torch.stack(back_alignment_weights, dim=1),
                    best_indices,
                )
            )

        return outputs

    @torch.jit.script_method
    def _get_output_steps_to_beam_indices(
        self, end_state: Tensor, beam_prev_indices: Tensor
    ) -> List[int]:
        """
        Returns a mapping from each output position and the beam index that was
        picked from the beam search results.
        """
        present_position = int(end_state[1])
        beam_index = int(end_state[2])
        beam_indices = torch.jit.annotate(List[int], [])
        while present_position >= 0:
            beam_indices.insert(0, beam_index)
            beam_index = int(beam_prev_indices[present_position][beam_index])
            present_position = present_position - 1
        return beam_indices

    @torch.jit.script_method
    def _add_to_end_states(
        self, end_states: List[Tensor], min_score: float, state: Tensor, min_index: int
    ) -> Tuple[List[Tensor], float, int]:
        """
        Maintains a list of atmost `nbest` highest end states
        """
        if len(end_states) < self.nbest:
            end_states.append(state)
            # keep min_score and min_index updated
            if float(state[0]) <= min_score:
                min_score = float(state[0])
                min_index = len(end_states) - 1
        elif bool(state[0] > min_score):
            # replace worst hypo with the new one
            end_states[min_index] = state
            # find new worst hypo, keep min_score and min_index updated
            min_index = -1
            min_score = float("inf")
            for idx in range(len(end_states)):
                s = end_states[idx]
                if bool(float(s[0]) <= min_score):
                    min_index = idx
                    min_score = float(s[0])
        return end_states, min_score, min_index

    @torch.jit.script_method
    def _get_all_end_states(
        self,
        beam_tokens: Tensor,
        beam_scores: Tensor,
        beam_prev_indices: Tensor,
        num_steps: int,
    ) -> Tensor:
        """
        Return all end states and hypothesis scores for those end states.
        """
        min_score = float("inf")
        min_index = -1
        end_states = torch.jit.annotate(List[Tensor], [])
        prev_hypo_is_finished = torch.zeros(self.beam_size).byte()

        position = 1
        while bool(position <= num_steps):
            hypo_is_finished = torch.zeros(self.beam_size).byte()

            for hyp_index in range(self.beam_size):
                prev_pos = beam_prev_indices[position][hyp_index]
                hypo_is_finished[hyp_index] = prev_hypo_is_finished[prev_pos]

                # If hypothesis was completed in the previous index,
                # then just continue
                if bool(hypo_is_finished[hyp_index] == 0):
                    # If the present token is EOS or we have reached max_length
                    # then hypothesis is complete
                    if bool(
                        beam_tokens[position][hyp_index] == self.eos_token_id
                    ) or bool(position == num_steps):

                        if bool(self.stop_at_eos):
                            hypo_is_finished[hyp_index] = 1

                        hypo_score = float(beam_scores[position][hyp_index])
                        if bool(self.length_penalty != 0):
                            hypo_score = hypo_score / float(position) ** float(
                                self.length_penalty
                            )

                        end_states, min_score, min_index = self._add_to_end_states(
                            end_states,
                            min_score,
                            torch.tensor(
                                [hypo_score, float(position), float(hyp_index)]
                            ),
                            min_index,
                        )

            prev_hypo_is_finished = hypo_is_finished
            position = position + 1

        end_states = torch.stack(end_states)

        _, sorted_end_state_indices = end_states[:, 0].sort(dim=0, descending=True)
        end_states = end_states[sorted_end_state_indices, :]
        return end_states

    @torch.jit.script_method
    def _check_dimensions(
        self,
        beam_tokens: Tensor,
        beam_scores: Tensor,
        token_weights: Tensor,
        beam_prev_indices: Tensor,
        num_steps: int,
    ) -> None:

        assert (
            beam_tokens.size(1) == self.beam_size
        ), "Dimension of beam_tokens : {} and beam size : {} are not consistent".format(
            beam_tokens.size(), self.beam_size
        )
        assert beam_scores.size(1) == self.beam_size, (
            "Dimension of beam_scores : {} and beam size : {} "
            "are not consistent".format(beam_scores.size(), self.beam_size)
        )
        assert token_weights.size(1) == self.beam_size, (
            "Dimension of token_weights : {} and beam size : {} "
            "are not consistent".format(token_weights.size(), self.beam_size)
        )
        assert (
            beam_prev_indices.size(1) == self.beam_size
        ), "Dimension of beam_prev_indices : {} and beam size : {} "
        "are not consistent".format(beam_prev_indices.size(), self.beam_size)

        assert beam_tokens.size(0) <= num_steps + 1, (
            "Dimension of beam_tokens : {} and num_steps : {} "
            "are not consistent".format(beam_tokens.size(), num_steps)
        )
        assert beam_scores.size(0) <= num_steps + 1, (
            "Dimension of beam_scores : {} and num_steps : {} "
            "are not consistent".format(beam_scores.size(), num_steps)
        )
        assert token_weights.size(0) <= num_steps + 1, (
            "Dimension of token_weights : {} and num_steps : {} "
            "are not consistent".format(token_weights.size(), num_steps)
        )
        assert beam_prev_indices.size(0) <= num_steps + 1, (
            "Dimension of beam_prev_indices : {} and num_steps : {} "
            "are not consistent".format(beam_prev_indices.size(), num_steps)
        )
