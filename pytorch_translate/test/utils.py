#!/usr/bin/env python3

import codecs
import tempfile

import numpy as np
import torch
from fairseq import data
from pytorch_translate import (
    dictionary as pytorch_translate_dictionary,
    vocab_constants,
)


class ModelParamsDict:
    def __init__(self, transformer=False, **kwargs):
        # Model params
        if transformer:
            self.arch = "ptt_transformer"
            self.encoder_embed_dim = 10
            self.encoder_ffn_embed_dim = 16
            self.encoder_layers = 2
            self.encoder_attention_heads = 2
            self.decoder_embed_dim = 10
            self.decoder_ffn_embed_dim = 16
            self.decoder_layers = 2
            self.decoder_attention_heads = 2
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
            self.decoder_embed_dim = 10
            self.decoder_embed_path = None
            self.decoder_freeze_embed = False
            self.decoder_hidden_dim = 10
            self.decoder_out_embed_dim = 5
            self.decoder_out_embed_path = None
            self.decoder_layers = 2
            self.dropout = 0
            self.decoder_dropout_in = 0
            self.decoder_dropout_out = 0
            self.attention_type = "dot"
            self.residual_level = None
            self.averaging_encoder = False
            self.cell_type = "lstm"
            self.sequence_lstm = False
            self.decoder_tie_embeddings = False
            self.language_model_only = False
        # Training params
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
        self.word_dropout_params = None
        self.distributed_world_size = 1
        self.seed = 1
        self.left_pad_source = "True"
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
