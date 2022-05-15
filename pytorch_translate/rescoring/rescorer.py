#!/usr/bin/env python3

import argparse
import pickle
from enum import Enum

import torch
from fairseq.scoring import bleu
from pytorch_translate import hybrid_transformer_rnn, utils  # noqa
from pytorch_translate.rescoring.model_scorers import (
    LMScorer,
    R2LModelScorer,
    ReverseModelScorer,
    SimpleModelScorer,
)
from pytorch_translate.research.rescore import cloze_transformer_model  # noqa
from pytorch_translate.tasks import pytorch_translate_task  # noqa
from tqdm import tqdm


class FeatureList(Enum):
    L2R_MODEL_SCORE = 0
    R2L_MODEL_SCORE = 1
    REVERSE_MODEL_SCORE = 2
    LM_SCORE = 3
    CLOZE_SCORE = 4


class Rescorer:
    """Reranks n-best hypotheses based on extra models and parameters"""

    def __init__(self, args, forward_task=None, models=None):
        """models = {'l2r_model': {'model': model, 'task': task}, ...}"""
        self.args = args
        if models is None:
            models = {}
        self.l2r_model_scorer = None
        if args.l2r_model_path or models.get("l2r_model", None):
            self.l2r_model_scorer = SimpleModelScorer(
                args, args.l2r_model_path, models.get("l2r_model", None), forward_task
            )

        self.r2l_model_scorer = None
        if args.r2l_model_path or models.get("r2l_model", None):
            self.r2l_model_scorer = R2LModelScorer(
                args, args.r2l_model_path, models.get("r2l_model", None), forward_task
            )

        self.reverse_model_scorer = None
        if args.reverse_model_path or models.get("reverse_model", None):
            self.reverse_model_scorer = ReverseModelScorer(
                args,
                args.reverse_model_path,
                models.get("reverse_model", None),
                forward_task,
            )

        self.lm_scorer = None
        if args.lm_model_path or models.get("lm_model", None):
            self.lm_scorer = LMScorer(
                args, args.lm_model_path, models.get("lm_model", None), forward_task
            )

        self.cloze_transformer_scorer = None
        if args.cloze_transformer_path or models.get("cloze_model", None):
            self.cloze_transformer_scorer = SimpleModelScorer(
                args,
                args.cloze_transformer_path,
                models.get("cloze_model", None),
                forward_task,
            )

    def score(self, src_tokens, hypos):
        """run models and compute scores based on p(y), p(x|y) etc."""

        scores = torch.zeros((len(hypos), len(FeatureList)), dtype=torch.float)

        self.compute_l2r_model_scores(src_tokens, hypos, scores)
        self.compute_r2l_model_scores(src_tokens, hypos, scores)
        self.compute_reverse_model_scores(src_tokens, hypos, scores)
        self.compute_lm_scores(src_tokens, hypos, scores)
        self.compute_cloze_transformer_scores(src_tokens, hypos, scores)

        return scores

    def compute_l2r_model_scores(self, src_tokens, hypos, scores):
        if not self.l2r_model_scorer:
            return
        l2r_scores = self.l2r_model_scorer.score(src_tokens, hypos)
        scores[:, FeatureList.L2R_MODEL_SCORE.value] = l2r_scores[:]

    def compute_r2l_model_scores(self, src_tokens, hypos, scores):
        if not self.r2l_model_scorer:
            return
        r2l_scores = self.r2l_model_scorer.score(src_tokens, hypos)
        scores[:, FeatureList.R2L_MODEL_SCORE.value] = r2l_scores[:]

    def compute_reverse_model_scores(self, src_tokens, hypos, scores):
        """computes p(x|y) for each hypothesis."""
        if not self.reverse_model_scorer:
            return

        scores[
            :, FeatureList.REVERSE_MODEL_SCORE.value
        ] = self.reverse_model_scorer.score(src_tokens, hypos)

    def compute_lm_scores(self, src_tokens, hypos, scores):
        """computes p(x|y) for each hypothesis."""
        if not self.lm_scorer:
            return

        lm_scores = self.lm_scorer.score(src_tokens, hypos)
        scores[:, FeatureList.LM_SCORE.value] = lm_scores[:]

    def compute_cloze_transformer_scores(self, src_tokens, hypos, scores):
        if not self.cloze_transformer_scorer:
            return

        cloze_scores = self.cloze_transformer_scorer.score(src_tokens, hypos)
        scores[:, FeatureList.CLOZE_SCORE.value] = cloze_scores[:]


def add_args(parser):
    """add rescorer specific arguments usable for training"""
    parser.add_argument(
        "--l2r-model-path",
        default=None,
        type=str,
        help=("Provide a path for the l2r rescoring model"),
    )
    parser.add_argument(
        "--l2r-model-weight",
        default=1.0,
        type=float,
        help=("Provide a weight for the l2r rescoring model"),
    )
    parser.add_argument(
        "--r2l-model-path",
        default=None,
        type=str,
        help=("Provide a path for the r2l rescoring model"),
    )
    parser.add_argument(
        "--r2l-model-weight",
        default=1.0,
        type=float,
        help=("Provide a weight for the r2l rescoring model"),
    )
    parser.add_argument(
        "--reverse-model-path",
        default=None,
        type=str,
        help=("Provide a path for the reverse rescoring model"),
    )
    parser.add_argument(
        "--reverse-model-weight",
        default=1.0,
        type=float,
        help=("Provide a weight for the reverse rescoring model"),
    )
    parser.add_argument(
        "--lm-model-path",
        default=None,
        type=str,
        help=("Provide a path for the language model rescoring model"),
    )
    parser.add_argument(
        "--lm-model-weight",
        default=1.0,
        type=float,
        help=("Provide a weight for the lm rescoring model"),
    )
    parser.add_argument(
        "--rescore-length-penalty",
        default=1.0,
        type=float,
        help=("Provide a weight for length penalty used in rescoring"),
    )
    parser.add_argument(
        "--cloze-transformer-path",
        default=None,
        type=str,
        help=("Provide a path for the cloze transformer rescoring model"),
    )
    parser.add_argument(
        "--cloze-transformer-weight",
        default=1.0,
        type=float,
        help=("Provide a weight for the cloze transformer model"),
    )


def add_args_rescore(parser):
    """add arguments only used in rescoring mode"""
    parser.add_argument(
        "--unk-reward",
        default=-1.0,
        type=float,
        help=(
            "Value to add to (log-prob) score for UNK tokens. "
            "Value < 0 (the usual case) encourages fewer UNKs, while > 0 "
            "encourages more UNKs."
        ),
    )
    parser.add_argument(
        "--append-eos-to-source",
        default=False,
        type=bool,
        help=("If true, append EOS to source sentences"),
    )
    parser.add_argument(
        "--batch-size", default=1, type=int, help="batch size for rescoring"
    )
    parser.add_argument(
        "--translation-info-export-path",
        default=None,
        type=str,
        help=("Optional path to save translation info output in pickled format"),
    )
    parser.add_argument(
        "--scores-info-export-path",
        default=None,
        type=str,
        help=("Optional path to save score output in pickled format"),
    )


def combine_weighted_scores(scores, weights, src_len, tgt_len, lenpen):
    """Combines scores from different models and returns

    Args:
        scores: scores for each feature and hypo [num_of_hypos, num_of_features]
        weights: weights for each feature [num_of_features]
        src_len: number of source sentence tokens
        tgt_len: list of target sentence tokens lengths [num_of_hypos]
        lenpen: float representing length penalty

    Returns:
        weighted_scores: one unified score for each hypothesis [num_of_hypos]
    """
    weighted_scores = scores.clone()
    weighted_scores[:, FeatureList.L2R_MODEL_SCORE.value] /= tgt_len**lenpen
    weighted_scores[:, FeatureList.R2L_MODEL_SCORE.value] /= tgt_len**lenpen
    weighted_scores[:, FeatureList.REVERSE_MODEL_SCORE.value] /= src_len**lenpen
    weighted_scores[:, FeatureList.LM_SCORE.value] /= tgt_len**lenpen
    weighted_scores[:, FeatureList.CLOZE_SCORE.value] /= tgt_len**lenpen

    weighted_scores *= torch.tensor(weights)
    # convert [num_of_hypos, num_of_features] to [num_of_hypos] and return
    return weighted_scores.sum(dim=1)


def find_top_tokens(args, trans_batch_info, rescorer, pad):
    """Rescore translations and combine weights to find top hypo tokens"""
    len_src_tokens = [len(trans_info["src_tokens"]) for trans_info in trans_batch_info]
    bsz = len(trans_batch_info)
    src_tokens = torch.zeros(bsz, max(len_src_tokens)).fill_(pad).long().cuda()
    for i in range(bsz):
        src_tokens[i, : len_src_tokens[i]] = (
            trans_batch_info[i]["src_tokens"].view(1, -1).long().cuda()
        )
    hypos = [hypo for trans_info in trans_batch_info for hypo in trans_info["hypos"]]

    scores = rescorer.score(src_tokens, hypos)

    # Prepare all the weights and call combine weighted scores
    weights = [
        args.l2r_model_weight,
        args.r2l_model_weight,
        args.reverse_model_weight,
        args.lm_model_weight,
        args.cloze_transformer_weight,
    ]
    bsz, src_len = src_tokens.size()
    beam_size = len(hypos) // bsz
    hypos = [hypos[i * beam_size : (i + 1) * beam_size] for i in range(bsz)]
    scores_to_export = []
    top_hypos = []
    for i in range(bsz):
        score = scores[i * beam_size : (i + 1) * beam_size, :]
        tgt_len = torch.tensor(
            [len(hypo["tokens"]) for hypo in hypos[i]], dtype=torch.float
        )
        combined_scores = combine_weighted_scores(
            score, weights, len_src_tokens[i], tgt_len, args.rescore_length_penalty
        )
        top_index = combined_scores.max(0)[1]
        scores_to_export.append(
            {
                "hypos": [hypo["tokens"].cpu().tolist() for hypo in hypos[i]],
                "target_tokens": trans_batch_info[i]["target_tokens"].cpu().numpy(),
                "scores": score.detach().numpy(),
                "src_len": len_src_tokens[i],
                "tgt_len": tgt_len.cpu().numpy(),
            }
        )
        top_hypos.append(hypos[i][top_index]["tokens"])
    return top_hypos, scores_to_export


def main():
    parser = argparse.ArgumentParser(
        description=("Rescore generated hypotheses with extra models")
    )
    add_args(parser)
    add_args_rescore(parser)
    args = parser.parse_args()

    assert (
        args.translation_info_export_path is not None
    ), "--translation_info_export_path is required for rescoring"

    assert args.l2r_model_path is not None, "Rescoring needs forward model"

    _, _, forward_task = utils.load_diverse_ensemble_for_inference(
        [args.l2r_model_path]
    )
    rescorer = Rescorer(args, forward_task)
    dst_dict = forward_task.tgt_dict
    base_bleu_scorer = bleu.Scorer(dst_dict.pad(), dst_dict.eos(), dst_dict.unk())
    rescoring_bleu_scorer = bleu.Scorer(
        bleu.BleuConfig(
            pad=dst_dict.pad(),
            eos=dst_dict.eos(),
            unk=dst_dict.unk(),
        )
    )

    with open(args.translation_info_export_path, "rb") as file:
        translation_info_list = pickle.load(file)

    scores_to_export_list = []
    trans_batch_info = []
    for k in tqdm(range(0, len(translation_info_list), args.batch_size)):
        trans_batch_info = translation_info_list[k : k + args.batch_size]
        for j in range(len(trans_batch_info)):
            trans_batch_info[j]["hypos"] = [
                {"score": hypo["score"], "tokens": hypo["tokens"].cuda()}
                for hypo in trans_batch_info[j]["hypos"]
            ]
        top_tokens, scores_to_export = find_top_tokens(
            args, trans_batch_info, rescorer, dst_dict.pad()
        )
        if args.scores_info_export_path is not None:
            scores_to_export_list += scores_to_export

        for i, trans_info in enumerate(trans_batch_info):
            base_bleu_scorer.add(
                trans_info["target_tokens"].int().cpu(),
                trans_info["hypos"][0]["tokens"].int().cpu(),
            )
            rescoring_bleu_scorer.add(
                trans_info["target_tokens"].int().cpu(), top_tokens[i].int().cpu()
            )
        trans_batch_info = []

    print("| Base ", base_bleu_scorer.result_string())
    print("| Rescoring ", rescoring_bleu_scorer.result_string())

    if args.scores_info_export_path is not None:
        with open(args.scores_info_export_path, "wb") as file:
            pickle.dump(scores_to_export_list, file)


if __name__ == "__main__":
    main()
