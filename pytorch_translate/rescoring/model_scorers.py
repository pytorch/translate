#!/usr/bin/env python3

import math

import torch
from pytorch_translate import utils as pytorch_translate_utils


class SimpleModelScorer(object):
    """Rescores source and target tokens based on a model"""

    def __init__(self, args, model_path):
        self.args = args
        # TODO (T40938917): Allow loading of multiple rescoring models
        rescoring_model, rescoring_model_arg, rescoring_task = pytorch_translate_utils.load_diverse_ensemble_for_inference(
            [model_path]
        )
        self.task = rescoring_task
        self.model = rescoring_model[0]
        self.model.eval()

        use_cuda = torch.cuda.is_available() and not self.args.cpu
        if use_cuda:
            self.model.cuda()

    def convert_hypos_to_tgt_tokens(self, hypos):
        """
        hypos contains target tokens for the original model for each hypothesis.
        we convert them to a tensor, also add eos token to the beginning,
        0's to the end, so that we can run model encoder and decoder on them
        """
        max_tgt_len = max(len(hypo["tokens"]) for hypo in hypos)
        tgt_tokens = torch.zeros(len(hypos), max_tgt_len + 1, dtype=torch.int)
        eos = self.task.target_dictionary.eos()
        tgt_tokens[:, 0] = torch.tensor(eos)

        for i, hypo in enumerate(hypos):
            start = 1
            end = start + len(hypo["tokens"])
            tgt_tokens[i, start:end] = hypo["tokens"]

        return tgt_tokens.long()

    def reverse_tgt_tokens(self, tgt_tokens):
        """
        tgt_tokens has 0 paddings at the end since they are batched.
        while reversing, we should roll first to keep 0's at the right.

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

        for i, row in enumerate(tgt_tokens):
            zero_count = len(row) - row.nonzero().size()[0]
            reversed_tgt_tokens[i] = reversed(roll(row, zero_count))

        return reversed_tgt_tokens

    def prepare_encoder_inputs(self, src_tokens):
        src_length = len(src_tokens)
        src_tokens = src_tokens.unsqueeze(
            0
        )  # we add a dimension because its not batched yet
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
        logprobs[:, :, pad] = -math.inf  # never select pad

        possible_translation_tokens = decoder_out[2]
        unk_pos = torch.nonzero(possible_translation_tokens == unk)
        if unk_pos.size()[0] != 0:
            # only add unk_reward if unk index appears in possible_translation_tokens
            unk_index = unk_pos[0][0]
            logprobs[:, :, unk_index] += args.unk_reward

            return logprobs, possible_translation_tokens

    def compute_scores(self, tgt_tokens, logprobs, possible_translation_tokens):
        """ Extract scores from logprobs for each hypothesis
        """
        pad = self.task.target_dictionary.pad()

        tgt_tokens = tgt_tokens[:, 1:]  # get rid of initial eos token
        hypos_tokens_probs = torch.zeros(tgt_tokens.shape)
        for i, hypo_tokens in enumerate(tgt_tokens):
            for j, hypo_token in enumerate(hypo_tokens):
                if hypo_token != pad:
                    hypos_tokens_probs[i][j] = logprobs[i][j][
                        (possible_translation_tokens == hypo_token).nonzero()
                    ]

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
