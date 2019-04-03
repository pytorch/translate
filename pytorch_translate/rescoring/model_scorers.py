#!/usr/bin/env python3

import torch
from pytorch_translate import utils


class SimpleModelScorer(object):
    """ Rescores source and target tokens based on a model"""

    def __init__(self, args, model_path, original_task):
        """ Initialize a rescorer model

        Args:
          args: model arguments
          model_path: checkpoint path for rescoring model
          original_task: original task is required to map the differences
            between the original model and rescoring model. currently, we pass
            original source tokens and hypotheses from the original model
            to rescoring model. therefore, this class needs to know how to map
            original model tokens to rescoring model tokens
        """
        self.args = args
        # TODO (T40938917): Allow loading of multiple rescoring models
        (
            rescoring_model,
            rescoring_model_arg,
            rescoring_task,
        ) = utils.load_diverse_ensemble_for_inference([model_path])
        self.task = rescoring_task  # e.g p(y), p(x|y) etc.
        self.original_task = original_task  # p(y|x)
        self.model = rescoring_model[0]
        self.model.eval()

        if not self.args.cpu:
            utils.maybe_cuda(self.model)

    def convert_hypos_to_tgt_tokens(self, hypos):
        """
        hypos is a list of hypotheses containing elements of type dict.
        each hypothesis dictionary contains target tokens.
        we convert these target tokens to a tensor, also eos token to the
        left and right, and padding to the end.
        """
        max_tgt_len = max(len(hypo["tokens"]) for hypo in hypos)
        pad = self.original_task.target_dictionary.pad()
        tgt_tokens = torch.full(
            (len(hypos), max_tgt_len + 1), fill_value=pad, dtype=torch.long
        )
        eos = self.original_task.target_dictionary.eos()
        tgt_tokens[:, 0] = torch.tensor(eos)

        for i, hypo in enumerate(hypos):
            start = 1
            end = start + len(hypo["tokens"])
            tgt_tokens[i, start:end] = hypo["tokens"]

        return tgt_tokens

    def reverse_tgt_tokens(self, tgt_tokens):
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

        pad = self.original_task.tgt_dict.pad()
        for i, row in enumerate(tgt_tokens):
            pad_count = sum(row == pad)
            reversed_tgt_tokens[i] = reversed(roll(row, int(pad_count)))

        return reversed_tgt_tokens

    def prepare_encoder_inputs(self, src_tokens):
        src_length = len(src_tokens)
        src_tokens = src_tokens.unsqueeze(0)  # batch dimension
        return (src_tokens, [src_length])

    def encode(self, args, encoder_inputs):
        assert (
            encoder_inputs[0] == self.task.target_dictionary.eos()
        ).sum() == 0, "Encoder doesn't expect eos tokens as input"

        encoder_out = self.model.encoder(*encoder_inputs)
        return [encoder_out]

    def decode(self, args, model, encoder_outs, tgt_tokens):
        """ Run model decoder on tgt_tokens and encoder_outputs

        Args:
          args: model arguments
          model: given rescoring model
          encoder_outs: encoder output. list(tuple([[input_length, batch_size,
            hidden_dim], [batch_size, input_length]))
          tgt_tokens: target tokens to be rescored. target tokens are expected
            to start with eos to signal start of the sentence, and expected to
            to end with eos to score eos. therefore target_size should be equal
            to number_of_target_tokens + 2, and should return
            number_of_target_tokens + 1 output. [batch_size, target_length]

        Returns:
          logprobs: log probabilities for each tgt token [batch_size,
            target_length, vocab_size]

        Raises:
          ValueError: If there is a problem with input.
          * If tgt_tokens don't start and end with eos.
        """
        eos = self.task.target_dictionary.eos()
        pad = self.task.target_dictionary.pad()
        unk = self.task.target_dictionary.unk()

        if (tgt_tokens == eos).sum() != 2 * tgt_tokens.size()[0]:
            raise ValueError("Each target should have 2 eos tokens")

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
        assert (
            len(decoder_out) == 2 or decoder_out[2] is None
        ), "Rescoring doesn't work with vocab reduction"

        logprobs = model.get_normalized_probs(decoder_out, log_probs=True)
        logprobs += args.word_reward
        logprobs[:, :, eos] -= args.word_reward
        logprobs[:, :, unk] += args.unk_reward
        logprobs[:, :, pad] = 0
        return logprobs

    def compute_scores(self, tgt_tokens, logprobs):
        """ logprobs have the log probabilities for each possible token
        for each hypothesis. here, we extract the logprobs matching the
        target tokens.
        """
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
        logprobs = self.decode(self.args, self.model, encoder_outs, tgt_tokens)
        hypos_scores = self.compute_scores(tgt_tokens, logprobs)

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


class ReverseModelScorer(SimpleModelScorer):
    """
    Scores p(x|y) with a reverse model and switching src and tgt sentences
    """

    def __init__(self, args, model_path, original_task):
        super().__init__(args, model_path, original_task)

    def prepare_inputs(self, src_tokens, hypo):
        """
        For reverse model, we need to switch src_tokens and tgt_tokens.
        We also make sure source is reversed if original task vs new task
        has different reverse_source settings.
        """
        eos = self.task.target_dictionary.eos()

        # Map token ids from original dictionary to reverse model dictionary
        src_string = self.original_task.src_dict.string(src_tokens)
        src_tokens_mapped = self.task.tgt_dict.encode_line(
            src_string, add_if_not_exist=False
        )

        tgt_string = self.original_task.tgt_dict.string(hypo["tokens"])
        tgt_tokens_mapped = self.task.src_dict.encode_line(
            tgt_string, add_if_not_exist=False
        )

        # Swap target and source tokens with necessary modifications
        tgt_tokens = (
            torch.cat(
                (
                    torch.tensor([eos]).type_as(src_tokens_mapped),
                    reversed(src_tokens_mapped)
                    if self.task.args.reverse_source
                    != self.original_task.args.reverse_source
                    else src_tokens_mapped,
                ),
                dim=0,
            )
            .view(1, -1)
            .type_as(src_tokens)
        )
        src_tokens = tgt_tokens_mapped[:-1].type_as(src_tokens)

        encoder_inputs = self.prepare_encoder_inputs(src_tokens)
        return encoder_inputs, tgt_tokens


class LMScorer(SimpleModelScorer):
    def convert_hypos_to_tgt_tokens(self, hypos):
        """
        Converts target tokens from the translation model dictionary
        to language model dictionary
        """
        # TODO: (T41818693) Map translation model vs LM model differences
        # and come up with a solution
        max_tgt_len = max(len(hypo["tokens"]) for hypo in hypos)
        pad = self.task.dictionary.pad_index
        tgt_tokens = torch.full(
            (len(hypos), max_tgt_len), fill_value=pad, dtype=torch.long
        )

        for i, hypo in enumerate(hypos):
            tgt_string = self.original_task.tgt_dict.string(hypo["tokens"])
            tgt_mapped = self.task.dictionary.encode_line(
                tgt_string, add_if_not_exist=False
            )
            tgt_tokens[i, : len(tgt_mapped)] = tgt_mapped

        return tgt_tokens

    @torch.no_grad()
    def score(self, src_tokens, hypos):
        if self.model is None:
            return

        _, tgt_tokens = self.prepare_inputs(src_tokens, hypos)

        decoder_out = self.model.decoder(tgt_tokens)
        logprobs = self.model.get_normalized_probs(decoder_out, log_probs=True)
        hypos_tokens_probs = logprobs.gather(
            dim=2, index=tgt_tokens.unsqueeze(2)
        ).squeeze(2)

        pad = self.task.dictionary.pad_index
        hypos_tokens_probs = (tgt_tokens != pad).float() * hypos_tokens_probs

        hypos_scores = hypos_tokens_probs.sum(dim=1) / (hypos_tokens_probs != 0).sum(
            dim=1, dtype=torch.float
        )
        return hypos_scores
