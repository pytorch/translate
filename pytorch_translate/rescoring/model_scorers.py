#!/usr/bin/env python3

import torch
from pytorch_translate import utils


class SimpleModelScorer(object):
    """Rescores source and target tokens based on a model"""

    def __init__(self, args, model_path=None, model=None, forward_task=None):
        """Initialize a rescorer model

        Args:
          args: model arguments
          model_path: checkpoint path for rescoring model
        """
        # TODO (T40938917): Allow loading of multiple rescoring models
        # allow to create an empty scorer w/o model
        self.args = args
        self.forward_task = forward_task
        self.task = None
        self.model = None
        # Instantiate the model
        if model is not None:
            self.model = model["model"]
            self.task = model["task"]
        elif model_path:
            rescoring_model, _, task = utils.load_diverse_ensemble_for_inference(
                [model_path]
            )
            self.model = rescoring_model[0]
            self.task = task

        if self.model is not None:
            self.model.eval()
            # Turn off gradient computation in eval mode
            for param in self.model.parameters():
                param.requires_grad = False
            utils.maybe_cuda(self.model)

    def convert_hypos_to_tgt_tokens(self, hypos):
        """
        hypos is a list of hypotheses containing elements of type dict.
        each hypothesis dictionary contains target tokens.
        we convert these target tokens to a tensor, also eos token to the
        left and right, and padding to the end.
        """
        max_tgt_len = max(len(hypo["tokens"]) for hypo in hypos)
        pad = self.task.target_dictionary.pad()
        tgt_tokens = torch.full(
            (len(hypos), max_tgt_len + 1), fill_value=pad, dtype=torch.long
        )
        eos = self.task.target_dictionary.eos()
        tgt_tokens[:, 0] = torch.tensor(eos)

        for i, hypo in enumerate(hypos):
            start = 1
            end = start + len(hypo["tokens"])
            tgt_tokens[i, start:end] = hypo["tokens"]

        return tgt_tokens

    def encode(self, encoder_inputs):
        encoder_out = self.model.encoder(*encoder_inputs)
        return [encoder_out]

    def decode(self, args, model, encoder_outs, tgt_tokens):
        """Run model decoder on tgt_tokens and encoder_outputs

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
        unk = self.task.target_dictionary.unk()

        if (tgt_tokens == eos).sum() != 2 * tgt_tokens.size()[0]:
            raise ValueError("Each target should have 2 eos tokens")

        decoder_out = list(model.decoder(tgt_tokens, encoder_outs[0]))
        possible_translation_tokens = decoder_out[2] if len(decoder_out) == 3 else None

        logprobs = model.get_normalized_probs(decoder_out, log_probs=True)

        if possible_translation_tokens is not None:
            unk_pos = torch.nonzero(possible_translation_tokens == unk)
            if unk_pos.size()[0] != 0:
                # add unk_reward if unk appears in possible_translation_tokens
                unk_index = unk_pos[0][0]
                logprobs[:, :, unk_index] += self.args.unk_reward

        return logprobs, possible_translation_tokens

    def compute_scores(self, tgt_tokens, logprobs, possible_translation_tokens=None):
        """logprobs have the log probabilities for each possible token
        for each hypothesis. here, we extract the logprobs matching the
        target tokens.
        """

        def clean_tgt_tokens(tgt_tokens, possible_translation_tokens):
            for i, hypo_tokens in enumerate(tgt_tokens):
                for j, hypo_token in enumerate(hypo_tokens):
                    tgt_tokens[i][j] = (
                        possible_translation_tokens == hypo_token
                    ).nonzero()
            return tgt_tokens

        if possible_translation_tokens is not None:
            tgt_tokens = clean_tgt_tokens(tgt_tokens, possible_translation_tokens)

        tgt_tokens = tgt_tokens[:, 1:]  # get rid of initial eos token

        i, j = torch.meshgrid(
            torch.arange(0, tgt_tokens.size(0)).long(),
            torch.arange(0, tgt_tokens.size(1)).long(),
        )
        hypos_tokens_probs = torch.zeros(tgt_tokens.shape)
        hypos_tokens_probs = logprobs[:, :-1, :][i, j, tgt_tokens]
        pad = self.task.target_dictionary.pad()
        mask = 1.0 - tgt_tokens.eq(pad).float()
        hypos_scores = (hypos_tokens_probs * mask).sum(dim=1)
        return hypos_scores

    def prepare_inputs(self, src_tokens, hypos):
        beam_size = len(hypos) // len(src_tokens)
        bsz, src_length = src_tokens.size()
        src_tokens_expand = (
            src_tokens.unsqueeze(dim=1)
            .expand(-1, beam_size, -1)
            .contiguous()
            .view(bsz * beam_size, -1)
        )
        src_lengths = (
            torch.tensor(src_length).repeat(bsz * beam_size).type_as(src_tokens)
        )
        encoder_inputs = (src_tokens_expand, src_lengths)
        tgt_tokens = self.convert_hypos_to_tgt_tokens(hypos).type_as(src_tokens)
        return encoder_inputs, tgt_tokens

    def score_tokens(self, encoder_inputs, hypos):
        """calculate scores for arbitrary hypos given encoder_inputs
        input:
            hypos: a tensor of length bsz*beam_size
        output:
            hypos_scores: \sum log prob over the entire hypo
            encoder_outs: hidden vectors from encoder
                          (e.g., used to stabilize rl training)
            logprobs: log prob over all time-steps
                      (e.g., used for word-level rl training)
        """
        encoder_outs = self.encode(encoder_inputs)
        logprobs, possible_translation_tokens = self.decode(
            self.args, self.model, encoder_outs, hypos
        )
        hypos_scores = self.compute_scores(hypos, logprobs, possible_translation_tokens)
        return hypos_scores, encoder_outs, logprobs

    def score(self, src_tokens, hypos):
        """Rescores hypotheses based on a given model and input tokens.
        src_tokens: a tensor with size bsz x max_src_len
        hypos: a list with length of bsz * beam_size
        """
        if self.model is None:
            return

        # if cuda is available, we convert src_tokens to cuda. all other
        # tensors copy src_tokens's type (cpu or gpu)
        if torch.cuda.is_available():
            src_tokens = src_tokens.cuda()

        encoder_inputs, tgt_tokens = self.prepare_inputs(src_tokens, hypos)
        hypos_scores, _, _ = self.score_tokens(encoder_inputs, tgt_tokens)
        return hypos_scores


class R2LModelScorer(SimpleModelScorer):
    """
    R2L model works by reversing target tokens to right to left direction
    """

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

        pad = self.task.tgt_dict.pad()
        for i, row in enumerate(tgt_tokens):
            pad_count = torch.sum(row == pad)
            reversed_tgt_tokens[i] = reversed(roll(row, int(pad_count)))

        return reversed_tgt_tokens

    def prepare_inputs(self, src_tokens, hypos):
        bsz, src_length = src_tokens.size()
        beam_size = len(hypos) // bsz
        src_lengths = torch.tensor([src_length]).repeat(len(hypos)).type_as(src_tokens)
        src_tokens = (
            src_tokens.unsqueeze(1)
            .expand(-1, beam_size, -1)
            .contiguous()
            .view(-1, src_length)
        )

        encoder_inputs = (src_tokens, src_lengths)
        tgt_tokens = self.convert_hypos_to_tgt_tokens(hypos).type_as(src_tokens)
        tgt_tokens = self.reverse_tgt_tokens(tgt_tokens)

        return encoder_inputs, tgt_tokens


class ReverseModelScorer(SimpleModelScorer):
    """
    Scores p(x|y) with a reverse model and switching src and tgt sentences
    """

    def prepare_inputs(self, src_tokens, hypos):
        """
        For reverse model, we need to switch src_tokens and tgt_tokens.
        We also make sure source is reversed if original task vs new task
        has different reverse_source settings.
        """
        eos = self.task.target_dictionary.eos()

        # Prepare target tokens
        # Map token ids from original dictionary to reverse model dictionary
        src_string = self.forward_task.src_dict.string(src_tokens)
        src_tokens_mapped = self.task.tgt_dict.encode_line(
            src_string, add_if_not_exist=False
        )[
            :-1
        ]  # remove eos
        # Swap target and source tokens with necessary modifications
        tgt_tokens = (
            torch.cat(
                (
                    torch.tensor([eos]).type_as(src_tokens_mapped),
                    reversed(src_tokens_mapped)
                    if self.forward_task.args.reverse_source
                    else src_tokens_mapped,
                    torch.tensor([eos]).type_as(src_tokens_mapped),
                ),
                dim=0,
            )
            .view(1, -1)
            .type_as(src_tokens)
        )
        # In reverse model, tgt_tokens are repeated instead of source
        tgt_tokens = tgt_tokens.repeat(len(hypos), 1)

        # Prepare source tokens
        max_tgt_len = max(len(hypo["tokens"]) for hypo in hypos)
        if not self.args.append_eos_to_source:
            max_tgt_len -= 1  # no eos
        pad = self.task.target_dictionary.pad()
        src_tokens = torch.full(
            (len(hypos), max_tgt_len), fill_value=pad, dtype=torch.long
        ).type_as(tgt_tokens)

        src_lengths = torch.zeros(len(hypos)).type_as(src_tokens)
        for i, hypo in enumerate(hypos):
            tgt_string = self.forward_task.tgt_dict.string(hypo["tokens"])
            tgt_tokens_mapped = self.task.src_dict.encode_line(
                tgt_string, add_if_not_exist=False
            )
            if not self.args.append_eos_to_source:
                tgt_tokens_mapped = tgt_tokens_mapped[:-1]  # last token is eos

            src_lengths[i] = len(tgt_tokens_mapped)
            src_tokens[i, : len(tgt_tokens_mapped)] = (
                reversed(tgt_tokens_mapped)
                if self.task.args.reverse_source
                else tgt_tokens_mapped
            )

        encoder_inputs = (src_tokens, src_lengths)
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
            tgt_string = self.forward_task.tgt_dict.string(hypo["tokens"])
            tgt_mapped = self.task.dictionary.encode_line(
                tgt_string, add_if_not_exist=False
            )
            tgt_tokens[i, : len(tgt_mapped)] = tgt_mapped

        return tgt_tokens

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

        return hypos_tokens_probs.sum(dim=1)
