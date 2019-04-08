#!/usr/bin/env python3

from typing import Dict

import numpy as np
import torch
from fairseq.tasks import register_task
from pytorch_translate import constants, utils as pytorch_translate_utils
from pytorch_translate.data import (
    data as pytorch_translate_data,
    utils as data_utils,
    weighted_data,
)
from pytorch_translate.research.knowledge_distillation.teacher_score_data import (
    TeacherDataset,
)
from pytorch_translate.tasks.pytorch_translate_task import PytorchTranslateTask


@register_task(constants.KNOWLEDGE_DISTILLATION_TASK)
class PytorchKnowledgeDistillationTask(PytorchTranslateTask):
    def __init__(self, args, src_dict, tgt_dict, char_source_dict=None):
        super().__init__(
            args,
            src_dict=src_dict,
            tgt_dict=tgt_dict,
            char_source_dict=char_source_dict,
        )
        self.top_k_teacher_tokens = args.top_k_teacher_tokens

        # Load model ensemble from checkpoints
        self.teacher_models, _, _ = pytorch_translate_utils.load_diverse_ensemble_for_inference(
            args.teacher_path.split(":")
        )
        if torch.cuda.is_available():
            for teacher_model in self.teacher_models:
                teacher_model = pytorch_translate_utils.maybe_cuda(teacher_model)

        # Memoized scores for teacher models. By having this and gradually memoizing
        # the values, we prevent the teacher model from keeping recalculating the
        # teacher scores.
        self.top_k_teacher_scores: Dict[int, np.ndarray] = {}
        self.top_k_teacher_indices: Dict[int, np.ndarray] = {}

    @staticmethod
    def add_args(parser):
        PytorchTranslateTask.add_args(parser)

        """Add knowledge-distillation arguments to the parser."""
        parser.add_argument(
            "--teacher-path",
            metavar="FILE",
            help="path(s) to teacher model file(s) colon separated",
        )
        parser.add_argument(
            "--top-k-teacher-tokens",
            type=int,
            default=8,
            help=(
                "Incorporating only the top k words from the teacher model.",
                "We zero out all other possibilities and normalize the probabilities",
                "based on the K top element.",
                "If top-k-teacher-tokens=0, it backs up to the original way of",
                "enumerating all.",
            ),
        )

    def load_dataset(
        self, split, src_bin_path, tgt_bin_path, weights_file=None, is_train=False
    ):
        """
        Currently this method does not support character models.
        """
        corpus = pytorch_translate_data.ParallelCorpusConfig(
            source=pytorch_translate_data.CorpusConfig(
                dialect=self.args.source_lang, data_file=src_bin_path
            ),
            target=pytorch_translate_data.CorpusConfig(
                dialect=self.args.target_lang, data_file=tgt_bin_path
            ),
            weights_file=weights_file,
        )

        if self.args.log_verbose:
            print("Starting to load binarized data files.", flush=True)
        data_utils.validate_corpus_exists(corpus=corpus, split=split)

        dst_dataset = pytorch_translate_data.InMemoryNumpyDataset.create_from_file(
            corpus.target.data_file
        )
        src_dataset = pytorch_translate_data.InMemoryNumpyDataset.create_from_file(
            corpus.source.data_file
        )
        if is_train:
            self.datasets[split] = TeacherDataset(
                src=src_dataset,
                src_sizes=src_dataset.sizes,
                src_dict=self.src_dict,
                tgt=dst_dataset,
                tgt_sizes=dst_dataset.sizes,
                tgt_dict=self.tgt_dict,
                teacher_models=self.teacher_models,
                top_k_teacher_tokens=self.top_k_teacher_tokens,
                top_k_teacher_scores=self.top_k_teacher_scores,
                top_k_teacher_indices=self.top_k_teacher_indices,
                left_pad_source=False,
            )
        else:
            self.datasets[split] = weighted_data.WeightedLanguagePairDataset(
                src=src_dataset,
                src_sizes=src_dataset.sizes,
                src_dict=self.src_dict,
                tgt=dst_dataset,
                tgt_sizes=dst_dataset.sizes,
                tgt_dict=self.tgt_dict,
                weights=None,
                left_pad_source=False,
            )

        if self.args.log_verbose:
            print("Finished loading dataset", flush=True)

        print(f"| {split} {len(self.datasets[split])} examples")
