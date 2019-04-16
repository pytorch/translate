#!/usr/bin/env python3

import codecs
import os
import random
import tempfile
from typing import Any, Dict, Tuple

import numpy as np
import torch
from fairseq import data
from fairseq.trainer import Trainer
from pytorch_translate import (
    dictionary as pytorch_translate_dictionary,
    vocab_constants,
)
from pytorch_translate.tasks import pytorch_translate_task as tasks


class ModelParamsDict:
    def __init__(self, arch="rnn", **kwargs):
        # Model params
        if arch == "transformer":
            self.arch = "ptt_transformer"
            self.encoder_embed_dim = 10
            self.encoder_ffn_embed_dim = 16
            self.encoder_layers = 2
            self.encoder_attention_heads = 2
            self.decoder_embed_dim = 10
            self.decoder_ffn_embed_dim = 16
            self.decoder_layers = 2
            self.decoder_attention_heads = 2
        elif arch == "hybrid_transformer_rnn":
            self.arch = "hybrid_transformer_rnn"
            self.encoder_embed_dim = 6
            self.encoder_ffn_embed_dim = 16
            self.encoder_layers = 2
            self.encoder_attention_heads = 2
            self.decoder_embed_dim = 10
            self.decoder_lstm_units = 13
            self.decoder_layers = 2
            self.decoder_attention_heads = 2
        elif arch == "dual_decoder_kd":
            self.arch = "dual_decoder_kd"
            self.encoder_embed_dim = 10
            self.encoder_ffn_embed_dim = 16
            self.encoder_layers = 2
            self.encoder_attention_heads = 2
            self.decoder_embed_dim = 10
            self.decoder_ffn_embed_dim = 16
            self.decoder_layers = 2
            self.decoder_attention_heads = 2
            self.student_decoder_embed_dim = 5
            self.student_decoder_layers = 2
            self.student_decoder_attention_heads = 2
            self.student_decoder_lstm_units = 7
            self.student_decoder_out_embed_dim = 4
            self.student_decoder_reduced_attention_dim = 8
        elif arch == "hybrid_dual_decoder_kd":
            self.arch = "hybrid_dual_decoder_kd"
            self.encoder_embed_dim = 10
            self.encoder_ffn_embed_dim = 16
            self.encoder_layers = 2
            self.encoder_attention_heads = 2
            self.decoder_embed_dim = 10
            self.decoder_lstm_units = 16
            self.decoder_layers = 2
            self.decoder_attention_heads = 2
            self.student_decoder_embed_dim = 5
            self.student_decoder_layers = 2
            self.student_decoder_attention_heads = 2
            self.student_decoder_lstm_units = 7
            self.student_decoder_out_embed_dim = 4
            self.student_decoder_reduced_attention_dim = 8
        else:
            self.arch = "rnn"
            self.encoder_embed_dim = 10
            self.encoder_embed_path = None
            self.encoder_freeze_embed = False
            self.encoder_hidden_dim = 10
            self.encoder_layers = 2
            self.encoder_bidirectional = False
            self.encoder_dropout_in = 0
            self.encoder_dropout_out = 0
            self.encoder_context_embed = False
            self.decoder_embed_dim = 10
            self.decoder_embed_path = None
            self.decoder_freeze_embed = False
            self.decoder_hidden_dim = 10
            self.decoder_out_embed_dim = 5
            self.decoder_out_embed_path = None
            self.out_embed_norm = None
            self.decoder_layers = 2
            self.dropout = 0
            self.decoder_dropout_in = 0
            self.decoder_dropout_out = 0
            self.attention_type = "dot"
            self.attention_heads = 8
            self.first_layer_attention = False
            self.residual_level = None
            self.averaging_encoder = False
            self.cell_type = "lstm"
            self.sequence_lstm = False
            self.decoder_tie_embeddings = False
            self.language_model_only = False
        # Training params
        self.unk_only_char_encoding = False
        self.criterion = "cross_entropy"
        self.lr = [0.1]
        self.optimizer = "sgd"
        self.momentum = 0
        self.label_smoothing = None
        self.weight_decay = 0.0
        self.lr_scheduler = "fixed"
        self.force_anneal = 0
        self.lr_shrink = 0
        self.sentence_avg = True
        self.clip_norm = 5.0
        self.batch_size = 4
        self.vocab_reduction_params = None
        self.distributed_world_size = 1
        self.seed = 1
        self.left_pad_source = "False"
        self.fp16 = False
        self.cpu = None
        self.reverse_source = False
        self.append_eos_to_source = False
        self.word_reward = 0.0
        self.length_penalty = 0.0
        # Rescoring params
        self.enable_rescoring = False
        self.l2r_model_path = None
        self.l2r_model_weight = None
        self.enable_r2l_rescoring = False
        self.r2l_model_path = None
        self.r2l_model_weight = None
        self.enable_reverse_rescoring = False
        self.reverse_model_path = None
        self.reverse_model_weight = None
        self.enable_lm_rescoring = False
        self.lm_model_path = None
        self.lm_model_weight = None
        # Modified params
        for param, value in kwargs.items():
            assert hasattr(
                self, param
            ), f"Tried to specify value for nonexistent property {param}."
            self.__setattr__(param, value)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def dummy_dictionary(dummy_tokens=3, additional_token_list=None):
    """First adds the amount of dummy_tokens that you specify, then
    finally the additional_token_list, which is a list of string token values"""
    d = pytorch_translate_dictionary.Dictionary()
    for i in range(dummy_tokens):
        token = f"token_{i}"
        d.add_symbol(token)
    if additional_token_list is not None:
        for token in additional_token_list:
            d.add_symbol(token)
    d.finalize(padding_factor=-1)
    return d


def prepare_inputs(
    test_args, source_vocab_size=103, target_vocab_size=103, is_variable_seqlen=False
):
    # first 100 indices are reserved for special tokens
    src_dict = dummy_dictionary(dummy_tokens=source_vocab_size - 100)
    tgt_dict = dummy_dictionary(dummy_tokens=source_vocab_size - 100)

    def get_single_example(sample_id, src_sentence_length, tgt_sentence_length):
        non_special_start = 4
        example = {
            "id": sample_id,
            # Note: both source and target-side sentences are expected
            # to end in the EOS marker. LanguagePairDataset then:
            # (1) moves the EOS to the start of the target, for input feeding
            # (2) it also handles left (right) padding of the source (target)
            "source": torch.LongTensor(
                np.random.randint(
                    low=non_special_start,
                    high=len(src_dict.symbols),
                    size=src_sentence_length,
                ).tolist()
                + [src_dict.eos()]
            ),
            "target": torch.LongTensor(
                np.random.randint(
                    low=non_special_start,
                    high=len(tgt_dict.symbols),
                    size=tgt_sentence_length,
                ).tolist()
                + [tgt_dict.eos()]
            ),
        }
        return example

    min_sent_len = 7
    max_sent_len = 12
    fixed_tgt_length = 12
    dataset = TestDataset(
        [
            get_single_example(
                example_id,
                np.random.randint(low=min_sent_len, high=max_sent_len, size=1)
                if is_variable_seqlen
                else 10,
                fixed_tgt_length,
            )
            for example_id in range(test_args.batch_size)
        ]
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=test_args.batch_size,
        collate_fn=(
            lambda samples: data.language_pair_dataset.collate(
                samples, src_dict.pad(), src_dict.eos()
            )
        ),
    )
    data_iterator = iter(dataloader)
    return data_iterator, src_dict, tgt_dict


def create_dummy_extra_state(**kwargs):
    extra_state = {
        "epoch": 1,
        "batch_offset": 0,
        "val_loss": None,
        "start_time": 0,
        "last_bleu_eval": 0,
    }
    for param, value in kwargs.items():
        assert (
            param in extra_state
        ), f"Tried to specify value for nonexistent property {param}."
        extra_state[param] = value
    return extra_state


def create_lexical_dictionaries():
    lexical_dictionary_path = write_lines_to_temp_file(
        [
            "a A 0.7",
            "a B 0.3",
            "b C 0.1",
            "b D 0.8",
            "b E 0.1",
            "c A 0.3",
            "c B 0.4",
            "c C 0.3",
            "d D 0.4",
            "d E 0.3",
            "d A 0.2",
            "d B 0.1",
            "e C 1.0",
        ]
    )
    return [lexical_dictionary_path]


def create_pretrained_embed(dictionary, embed_dim):
    """Creates a dummy embedding file in the format accepted by fairseq. An
    embedding file has the following format: the first line has vocabulary size
    and dimension. The following lines contain word and space-separated
    embedding values.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932

    Arguments:
        dictionary (fairseq.data.dictionary.Dictionary): dictionary with
            sample tokens as entries
        embed_dim (int): embedding dimension to be generated

    Returns:
        Path to a text file with dummy embeddings in the format described above.
    """

    embed_weights = np.random.random((len(dictionary), embed_dim))
    pretrained_embed_path = write_lines_to_temp_file(
        ["{} {}".format(len(dictionary), embed_dim)]
        + [
            "{} {}".format(token, " ".join([str(val) for val in embedding]))
            for token, embedding in zip(dictionary.symbols, embed_weights)
        ]
    )
    return pretrained_embed_path, embed_weights


def create_test_text_files():
    src = write_lines_to_temp_file(
        [
            "srcA srcB srcC srcD",
            "srcA srcA srcB srcB srcC srcC",
            "srcA srcA srcA srcA srcB srcB srcB srcB",
            "srcA srcA srcA srcA srcA srcA srcA srcA srcA srcA",
        ]
    )
    trg = write_lines_to_temp_file(
        [
            "trgA trgA trgA trgA trgA trgA trgA trgA trgA trgA",
            "trgA trgA trgA trgA trgB trgB trgB trgB",
            "trgA trgA trgB trgB trgC trgC",
            "trgA trgB trgC trgD",
        ]
    )
    return src, trg


def create_test_numberized_data_files(src_ref, trg_ref, reverse_source=True):
    """
    Reformat ref from [[#, #, #], [#, #, #]] --> ["# # #", "# # #"]
    """
    if reverse_source:
        src_ref = [reversed(line) for line in src_ref]

    # during parsing
    src = write_lines_to_temp_file(
        [" ".join([str(ind) for ind in line]) for line in src_ref]
    )
    trg = write_lines_to_temp_file(
        [" ".join([str(ind) for ind in line]) for line in trg_ref]
    )
    return src, trg


def write_lines_to_temp_file(lines):
    temp_file_path = make_temp_file()
    with codecs.open(temp_file_path, "w", "utf-8") as temp_file:
        temp_file.write("\n".join(lines) + "\n")
    return temp_file_path


def make_temp_file():
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, dir="/tmp")
    temp_file_path = temp_file.name
    temp_file.close()
    return temp_file_path


def create_vocab_dictionaries():
    additional_special_tokens = (
        vocab_constants.MAX_SPECIAL_TOKENS
        - pytorch_translate_dictionary.Dictionary().nspecial
    )
    src_dict = dummy_dictionary(
        dummy_tokens=additional_special_tokens,
        additional_token_list=["a", "b", "c", "d", "e"],
    )
    tgt_dict = dummy_dictionary(
        dummy_tokens=additional_special_tokens,
        additional_token_list=["A", "B", "C", "D", "E"],
    )
    return src_dict, tgt_dict


def create_vocab_reduction_expected_array(
    src_dict, max_translation_candidates_per_word=1
):
    expected_translation_candidates = np.zeros(
        [len(src_dict), max_translation_candidates_per_word], dtype=np.int32
    )

    expected_translation_candidates[100][0] = 100
    expected_translation_candidates[101][0] = 103
    expected_translation_candidates[102][0] = 101
    expected_translation_candidates[103][0] = 103
    expected_translation_candidates[104][0] = 102

    return expected_translation_candidates


def gpu_train_step(test_args: ModelParamsDict) -> Tuple[Trainer, Dict[Any, Any]]:
    """Sets up inputs from test_args then executes a single train step. A train
    step always requires a GPU."""
    samples, src_dict, tgt_dict = prepare_inputs(test_args)
    task = tasks.DictionaryHolderTask(src_dict, tgt_dict)
    model = task.build_model(test_args)
    criterion = task.build_criterion(test_args)
    sample = next(samples)
    trainer = Trainer(test_args, task, model, criterion, dummy_batch=sample)
    logging_dict = trainer.train_step([sample])
    return trainer, logging_dict


def write_dummy_file(filename, num_examples, maxlen):
    rng_state = torch.get_rng_state()
    torch.manual_seed(0)
    data = torch.rand(num_examples * maxlen)
    data = 97 + torch.floor(26 * data).int()
    with open(filename, "w") as h:
        offset = 0
        for _ in range(num_examples):
            ex_len = random.randint(1, maxlen)
            ex_str = " ".join(map(chr, data[offset : offset + ex_len]))
            print(ex_str, file=h)
            offset += ex_len
    torch.set_rng_state(rng_state)


def create_dummy_data(data_dir, num_examples=100, maxlen=5):
    def _create_dummy_data(filename):
        write_dummy_file(os.path.join(data_dir, filename), num_examples, maxlen)

    _create_dummy_data("train.in")
    _create_dummy_data("train.out")
    _create_dummy_data("valid.in")
    _create_dummy_data("valid.out")
    _create_dummy_data("test.in")
    _create_dummy_data("test.out")


def create_dummy_multilingual_data(data_dir, num_examples=100, maxlen=5):
    def _create_dummy_data(filename):
        write_dummy_file(os.path.join(data_dir, filename), num_examples, maxlen)

    for src, tgt in [("xh", "en"), ("zu", "en")]:
        langpair = src + tgt
        _create_dummy_data(f"train.{langpair}.{src}")
        _create_dummy_data(f"train.{langpair}.{tgt}")
        _create_dummy_data(f"tune.{langpair}.{src}")
        _create_dummy_data(f"tune.{langpair}.{tgt}")
