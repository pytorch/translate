#!/usr/bin/env python3

import math

import torch
from pytorch_translate import utils as pytorch_translate_utils


def setup_rescoring(args):
    if args.rescoring_strategy is None or args.rescoring_model_path is None:
        return None

    # TODO (T40938917): Allow loading of multiple rescoring models
    rescoring_model, rescoring_model_arg, rescoring_task = pytorch_translate_utils.load_diverse_ensemble_for_inference(
        [args.rescoring_model_path]
    )
    return rescoring_model[0]


def prepare_encoder_inputs(src_tokens):
    src_length = len(src_tokens)
    src_tokens = src_tokens.unsqueeze(
        0
    )  # we add a dimension because its not batched yet
    return (src_tokens, [src_length])


def encode(args, model, encoder_inputs):
    model.eval()
    encoder_out = model.encoder(*encoder_inputs)
    return [encoder_out]


def convert_hypos_to_target_tokens_tensor(hypos):
    """
    hypos contains target tokens for the original model for each hypothesis.
    we convert them to a tensor, also add eos token to the beginning,
    0's to the end, so that we can run model encoder and decoder on them
    """
    max_tgt_len = max(len(hypo["tokens"]) for hypo in hypos)
    hypos_tokens = torch.zeros(len(hypos), max_tgt_len + 1, dtype=torch.int)
    hypos_tokens[:, 0] = torch.tensor(2)
    for i, hypo in enumerate(hypos):
        start = 1
        end = start + len(hypo["tokens"])
        hypos_tokens[i, start:end] = hypo["tokens"]
    return hypos_tokens.long()


def decode(args, model, task, encoder_outs, hypos_tokens):
    """ Run decoder with the same configurations with beam decoder
    """
    eos = task.target_dictionary.eos()
    pad = task.target_dictionary.pad()
    unk = task.target_dictionary.unk()

    reorder_indices = torch.arange(1).view(-1, 1).repeat(1, args.beam).view(-1)

    for i, encoder_out in enumerate(encoder_outs):
        # expand outputs for each example beam_size times
        encoder_outs[i] = model.encoder.reorder_encoder_out(
            encoder_out=encoder_out,
            new_order=reorder_indices.cuda()
            if encoder_out[0].is_cuda
            else reorder_indices,
        )

    decoder_out = list(model.decoder(hypos_tokens, encoder_outs[0]))
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


def get_scores(task, hypos_tokens, logprobs, possible_translation_tokens):
    """ Extract scores from logprobs for each hypothesis
    """
    pad = task.target_dictionary.pad()

    hypos_tokens = hypos_tokens[:, 1:]  # get rid of initial eos token
    hypos_tokens_probs = torch.zeros(hypos_tokens.shape)
    for i, hypo_tokens in enumerate(hypos_tokens):
        for j, hypo_token in enumerate(hypo_tokens):
            if hypo_token != pad:
                hypos_tokens_probs[i][j] = logprobs[i][j][
                    (possible_translation_tokens == hypo_token).nonzero()
                ]

    hypos_scores = hypos_tokens_probs.sum(dim=1)
    return hypos_scores


def run_rescoring(args, task, hypos, src_tokens, model):
    """ Rescores hypotheses based on a given model and input tokens.
    # TODO: (T40943663) Refactor rescoring into its own class
    # TODO: (T40961806) Proper testing for rescoring
    """
    if model is None:
        return

    hypos_tokens = convert_hypos_to_target_tokens_tensor(hypos)

    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        model.cuda()
        hypos_tokens = hypos_tokens.cuda()

    encoder_inputs = prepare_encoder_inputs(src_tokens)
    encoder_outs = encode(args, model, encoder_inputs)

    logprobs, possible_translation_tokens = decode(
        args, model, task, encoder_outs, hypos_tokens
    )

    hypos_scores = get_scores(task, hypos_tokens, logprobs, possible_translation_tokens)

    max_score_index = torch.max(hypos_scores, dim=0)[1]
    return hypos[max_score_index]["tokens"].int().cpu()
