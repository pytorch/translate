#!/usr/bin/env python3

import unittest

import numpy as np
import torch
import torch.nn.functional as F
from fairseq.data import language_pair_dataset
from pytorch_translate import char_source_model  # noqa
from pytorch_translate import rnn  # noqa
from pytorch_translate.research.knowledge_distillation import (
    collect_top_k_probs,
    dual_decoder_kd_loss,
    dual_decoder_kd_model,
    knowledge_distillation_loss,
)
from pytorch_translate.tasks import pytorch_translate_task as tasks
from pytorch_translate.test import utils as test_utils


class TestKnowledgeDistillation(unittest.TestCase):
    def test_topk_kd_loss(self):
        """
        Makes sure that we can build KD loss without problem.
        """
        test_args = test_utils.ModelParamsDict()
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        self.task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        sample = self._dummy_sample()
        model = self.task.build_model(test_args)
        net_output = model(**sample["net_input"])
        student_probs = model.get_normalized_probs(net_output, log_probs=True)
        # [bsz, seqlen, vocab] -> [bsz*seqlen, vocab]
        lprobs = student_probs.view(-1, student_probs.size(-1))

        teacher_model = self.task.build_model(test_args)
        teacher_probs = teacher_model.get_normalized_probs(net_output, log_probs=False)
        top_k_teacher_probs, indices = torch.topk(teacher_probs, k=3)
        top_k_teacher_probs_normalized = F.normalize(
            top_k_teacher_probs, p=1, dim=2
        ).detach()
        sample["top_k_scores"] = top_k_teacher_probs_normalized
        sample["top_k_indices"] = indices

        kd_criterion = knowledge_distillation_loss.KnowledgeDistillationCriterion(
            test_args, self.task
        )
        kd_loss, topk_probs = kd_criterion.get_kd_loss(sample, student_probs, lprobs)

        # asserting that the values are correctly inserted into teacher_probs.
        for row in range(indices.shape[0]):
            for col in range(indices.shape[1]):
                # testing if values are normalized.
                assert round(float(torch.sum(topk_probs[row][col][:])), 0) == 1.0
                for i, val in enumerate(indices[row][col]):
                    # testing if scattering is done correctly.
                    assert (
                        topk_probs[row][col][val]
                        == top_k_teacher_probs_normalized[row][col][i]
                    )

        topk_probs_flat = topk_probs.view(-1, topk_probs.size(-1))
        kd_loss = -torch.sum(topk_probs_flat * lprobs)
        assert kd_loss >= 0

    def test_dual_decoder_kd_loss(self):
        test_args = test_utils.ModelParamsDict(arch="dual_decoder_kd")
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        self.task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        sample = self._dummy_sample()
        model = self.task.build_model(test_args)

        test_args.kd_weight = 0.5
        test_args.label_smoothing = 0.1
        criterion = dual_decoder_kd_loss.DualDecoderCriterion(test_args, self.task)

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]

        encoder_out = model.encoder(src_tokens, src_lengths)
        student_output = model.student_decoder(prev_output_tokens, encoder_out)
        teacher_output = model.teacher_decoder(prev_output_tokens, encoder_out)

        teacher_loss, teacher_nll_loss, teacher_probs = criterion.compute_teacher_loss(
            model, teacher_output, sample, reduce=True
        )

        # probabilities for each label should sum to one
        assert all((teacher_probs.sum(dim=1) - 1.0).abs() < 1e-6)

        student_loss, student_nll_loss = criterion.compute_student_loss(
            model, student_output, sample, teacher_probs, reduce=True
        )

    def _dummy_sample(self, batch_size=3, input_seq_length=5, output_seq_length=4):

        output_sequence = torch.randint(
            low=self.task.dst_dict.nspecial,
            high=len(self.task.dst_dict),
            size=(batch_size, output_seq_length),
        ).long()
        eos_column = torch.LongTensor(
            [self.task.dst_dict.eos_index] * batch_size
        ).unsqueeze(1)
        prev_output_tokens = torch.cat([eos_column, output_sequence], dim=1)
        target_tokens = torch.cat([output_sequence, eos_column], dim=1)

        sample = {
            "net_input": {
                "src_tokens": torch.randint(
                    low=self.task.src_dict.nspecial,
                    high=len(self.task.src_dict),
                    size=(batch_size, input_seq_length),
                ).long(),
                "prev_output_tokens": prev_output_tokens,
                "src_lengths": torch.LongTensor([input_seq_length] * batch_size),
            },
            "target": target_tokens,
            "ntokens": target_tokens.numel(),
        }
        return sample

    def test_dual_decoder_args(self):
        test_args = test_utils.ModelParamsDict(arch="dual_decoder_kd")
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        self.task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        model = self.task.build_model(test_args)

        assert (
            model.encoder.transformer_embedding.embed_tokens.embedding_dim
            == test_args.encoder_embed_dim
        )
        assert (
            model.encoder.transformer_encoder_given_embeddings.layers[
                0
            ].fc1.out_features
            == test_args.encoder_ffn_embed_dim
        )
        assert (
            len(model.encoder.transformer_encoder_given_embeddings.layers)
            == test_args.encoder_layers
        )
        assert (
            model.encoder.transformer_encoder_given_embeddings.layers[
                0
            ].self_attn.num_heads
            == test_args.encoder_attention_heads
        )
        assert (
            model.teacher_decoder.embed_tokens.embedding_dim
            == test_args.decoder_embed_dim
        )
        assert (
            model.teacher_decoder.layers[0].fc1.out_features
            == test_args.decoder_ffn_embed_dim
        )
        assert len(model.teacher_decoder.layers) == test_args.decoder_layers
        assert (
            model.teacher_decoder.layers[0].self_attn.num_heads
            == test_args.decoder_attention_heads
        )
        assert (
            model.student_decoder.embed_tokens.embedding_dim
            == test_args.student_decoder_embed_dim
        )
        assert model.student_decoder.num_layers == test_args.student_decoder_layers
        assert (
            model.student_decoder.num_attention_heads
            == test_args.student_decoder_attention_heads
        )
        assert model.student_decoder.lstm_units == test_args.student_decoder_lstm_units
        assert (
            model.student_decoder.out_embed_dim
            == test_args.student_decoder_out_embed_dim
        )

    def test_collect_top_k_probs(self):
        test_args = test_utils.ModelParamsDict(arch="hybrid_transformer_rnn")
        _, src_dict, tgt_dict = test_utils.prepare_inputs(test_args)
        self.task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
        model = self.task.build_model(test_args)
        model.eval()

        binarized_source = test_utils.create_dummy_binarized_dataset()
        binarized_target = test_utils.create_dummy_binarized_dataset(append_eos=True)
        dataset = language_pair_dataset.LanguagePairDataset(
            src=binarized_source,
            src_sizes=binarized_source.sizes,
            src_dict=self.task.src_dict,
            tgt=binarized_target,
            tgt_sizes=binarized_target.sizes,
            tgt_dict=self.task.dst_dict,
            left_pad_source=False,
        )

        top_k_scores, top_k_indices = collect_top_k_probs.compute_top_k(
            task=self.task,
            models=[model],
            dataset=dataset,
            k=3,
            use_cuda=torch.cuda.is_available(),
            max_tokens=None,
            max_sentences=None,
            progress_bar_args=None,
        )

        batch = language_pair_dataset.collate(
            [dataset[0]],
            pad_idx=self.task.src_dict.pad(),
            eos_idx=self.task.src_dict.eos(),
            left_pad_source=False,
        )

        with torch.no_grad():
            net_output = model(**batch["net_input"])
            probs = model.get_normalized_probs(net_output, log_probs=False)

        top_probs, top_indices = torch.topk(probs[0, 0], k=3)

        np.testing.assert_array_equal(top_k_indices[0], top_indices.numpy())
        normalized_probs = (top_probs / top_probs.sum()).numpy()
        np.testing.assert_almost_equal(top_k_scores[0], normalized_probs)
