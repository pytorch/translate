#!/usr/bin/env python3

import torch
from pytorch_translate import utils


class SimpleModelScorer(object):
    """Rescores source and target tokens based on a model"""

    def __init__(self, args, model_path):
        self.args = args
        # TODO (T40938917): Allow loading of multiple rescoring models
        (
            rescoring_model,
            rescoring_model_arg,
            rescoring_task,
        ) = utils.load_diverse_ensemble_for_inference([model_path])
        self.task = rescoring_task
        self.model = rescoring_model[0]
        self.model.eval()

        if not self.args.cpu:
            utils.maybe_cuda(self.model)

    def convert_hypos_to_tgt_tokens(self, hypos):
        # TODO (T41749218): Add unit tests for rescoring
        """
        hypos is a list of hypotheses containing elements of type dict.
        each hypothesis dictionary contains target tokens.
        we convert these target tokens to a tensor, also eos token to the
        left, and padding to the right.
        """
        max_tgt_len = max(len(hypo["tokens"]) for hypo in hypos)
        pad = self.task.target_dictionary.pad()
        tgt_tokens = torch.full(
            (len(hypos), max_tgt_len + 1), fill_value=pad, dtype=torch.int
        )
        eos = self.task.target_dictionary.eos()
        tgt_tokens[:, 0] = torch.tensor(eos)

        for i, hypo in enumerate(hypos):
            start = 1
            end = start + len(hypo["tokens"])
            tgt_tokens[i, start:end] = hypo["tokens"]

        return tgt_tokens.long()

    def reverse_tgt_tokens(self, tgt_tokens):
        # TODO (T41749218): Add unit tests for rescoring
        """
        tgt_tokens has paddings to the right since they are batched.
        while reversing, we should roll first to keep paddings.

        Note:
            input:
                [1 2 3]
                [1 2 0]
                [1 0 0]
            output:
                [3 2 1]
                [2 1 0]
                [1 0 0]
        """
        reversed_tgt_tokens = torch.zeros_like(tgt_tokens)

        def roll(x, n):
            return torch.cat((x[-n:], x[:-n]))

        pad = self.task.tgt_dict.pad()
        for i, row in enumerate(tgt_tokens):
            pad_count = len(row) - sum(row == pad)
            reversed_tgt_tokens[i] = reversed(roll(row, pad_count))

        return reversed_tgt_tokens

    def prepare_encoder_inputs(self, src_tokens):
        src_length = len(src_tokens)
        src_tokens = src_tokens.unsqueeze(0)  # batch dimension
        return (src_tokens, [src_length])

    def encode(self, args, encoder_inputs):
        encoder_out = self.model.encoder(*encoder_inputs)
        return [encoder_out]

    def decode(self, args, model, encoder_outs, tgt_tokens):
        """ Run decoder with the same configurations with beam decoder
        """
        eos = self.task.target_dictionary.eos()
        pad = self.task.target_dictionary.pad()
        unk = self.task.target_dictionary.unk()

        reorder_indices = torch.arange(1).view(-1, 1).repeat(1, args.beam).view(-1)

        for i, encoder_out in enumerate(encoder_outs):
            # expand outputs for each example beam_size times
            encoder_outs[i] = model.encoder.reorder_encoder_out(
                encoder_out=encoder_out,
                new_order=reorder_indices.cuda()
                if encoder_out[0].is_cuda
                else reorder_indices,
            )

        decoder_out = list(model.decoder(tgt_tokens, encoder_outs[0]))
        assert len(decoder_out) == 3, "Rescoring only works with vocab reduction"

        logprobs = model.get_normalized_probs(decoder_out, log_probs=True)
        logprobs += args.word_reward
        logprobs[:, :, eos] -= args.word_reward
        logprobs[:, :, pad] = 0  # never select pad

        possible_translation_tokens = decoder_out[2]
        unk_pos = torch.nonzero(possible_translation_tokens == unk)
        if unk_pos.size()[0] != 0:
            # only add unk_reward if unk index appears in possible_translation_tokens
            unk_index = unk_pos[0][0]
            logprobs[:, :, unk_index] += args.unk_reward

            return logprobs, possible_translation_tokens

    def compute_scores(self, tgt_tokens, logprobs, possible_translation_tokens):
        """ logprobs have the log probabilities for each possible token
        for each hypothesis. here, we extract the logprobs matching the
        target tokens.
        """
        def clean_tgt_tokens(tgt_tokens, possible_translation_tokens):
            tgt_tokens_fixed = torch.zeros_like(tgt_tokens)
            for i, hypo_tokens in enumerate(tgt_tokens):
                for j, hypo_token in enumerate(hypo_tokens):
                    tgt_tokens_fixed[i][j] = (
                        possible_translation_tokens == hypo_token
                    ).nonzero()
            return tgt_tokens_fixed

        tgt_tokens = clean_tgt_tokens(tgt_tokens, possible_translation_tokens)
        tgt_tokens = tgt_tokens[:, 1:]  # get rid of initial eos token

        i, j = torch.meshgrid(
            torch.arange(0, tgt_tokens.size(0)).long(),
            torch.arange(0, tgt_tokens.size(1)).long(),
        )
        hypos_tokens_probs = torch.zeros(tgt_tokens.shape)
        hypos_tokens_probs = logprobs[:, :-1, :][i, j, tgt_tokens]
        hypos_scores = hypos_tokens_probs.sum(dim=1)
        return hypos_scores

    def prepare_inputs(self, src_tokens, hypos):
        encoder_inputs = self.prepare_encoder_inputs(src_tokens)
        tgt_tokens = self.convert_hypos_to_tgt_tokens(hypos).type_as(src_tokens)

        return encoder_inputs, tgt_tokens

    @torch.no_grad()
    def score(self, src_tokens, hypos):
        """ Rescores hypotheses based on a given model and input tokens.
        # TODO: (T40961806) Proper testing for rescoring
        """
        if self.model is None:
            return

        encoder_inputs, tgt_tokens = self.prepare_inputs(src_tokens, hypos)
        encoder_outs = self.encode(self.args, encoder_inputs)
        logprobs, possible_translation_tokens = self.decode(
            self.args, self.model, encoder_outs, tgt_tokens
        )
        hypos_scores = self.compute_scores(
            tgt_tokens, logprobs, possible_translation_tokens
        )

        return hypos_scores


class R2LModelScorer(SimpleModelScorer):
    """
    R2L model works by reversing target tokens to right to left direction
    """

    def prepare_inputs(self, src_tokens, hypos):
        encoder_inputs = self.prepare_encoder_inputs(src_tokens)
        tgt_tokens = self.convert_hypos_to_tgt_tokens(hypos).type_as(src_tokens)
        tgt_tokens = self.reverse_tgt_tokens(tgt_tokens)

        return encoder_inputs, tgt_tokens
