#!/usr/bin/env python3

import codecs
import numpy as np
import tempfile
import torch

from caffe2.python import workspace
import fblearner.flow.projects.langtech.translation.neural_mt.\
    seq2seq_model_caffe2 as seq2seq_model_caffe2
import fblearner.flow.projects.langtech.translation.neural_mt.\
    types as neural_mt_types
from fbtranslate import dictionary as fbtranslate_dictionary

from fairseq import data, utils

CAFFE2_MAX_SPECIAL_TOKENS = 100


class ModelParamsDict:

    def __init__(
        self,
        cell_type='lstm',
        encoder_bidirectional=False,
        encoder_freeze_embed=False,
        decoder_freeze_embed=False,
        clip_norm=5.0,
        sequence_lstm=False,
    ):
        # Model params
        self.arch = 'rnn'
        self.encoder_embed_dim = 10
        self.encoder_freeze_embed = encoder_freeze_embed
        self.encoder_hidden_dim = 10
        self.encoder_layers = 2
        self.encoder_bidirectional = encoder_bidirectional
        self.encoder_dropout_in = 0
        self.encoder_dropout_out = 0
        self.decoder_embed_dim = 10
        self.decoder_freeze_embed = decoder_freeze_embed
        self.decoder_hidden_dim = 10
        self.decoder_out_embed_dim = 5
        self.decoder_layers = 2
        self.dropout = 0
        self.decoder_dropout_in = 0
        self.decoder_dropout_out = 0
        self.attention_type = 'dot'
        self.residual_level = None
        self.averaging_encoder = False
        self.cell_type = cell_type
        self.sequence_lstm = sequence_lstm
        # Training params
        self.criterion = 'cross_entropy'
        self.lr = [0.1]
        self.optimizer = 'sgd'
        self.momentum = 0
        self.label_smoothing_epsilon = None
        self.weight_decay = 0.0
        self.lr_scheduler = 'fixed'
        self.force_anneal = 0
        self.lr_shrink = 0
        self.sentence_avg = True
        self.clip_norm = clip_norm
        self.batch_size = 4
        self.vocab_reduction_params = None
        self.word_dropout_params = None
        self.distributed_world_size = 1

    def caffe2_param_dict(self):
        assert len(self.lr) == 1
        assert self.weight_decay == 0
        assert self.attention_type == 'dot'
        assert self.label_smoothing_epsilon is None

        if self.encoder_bidirectional:
            assert self.encoder_hidden_dim % 2 == 0
            encoder_first_layer_dim = self.encoder_hidden_dim // 2
        else:
            encoder_first_layer_dim = self.encoder_hidden_dim

        # dropout omitted from equivalence testing due to stochasticity
        partial_model_params = dict(
            encoder_embedding_size=self.encoder_embed_dim,
            decoder_embedding_size=self.decoder_embed_dim,
            freeze_encoder_embeddings=self.encoder_freeze_embed,
            freeze_decoder_embeddings=self.decoder_freeze_embed,
            encoder_type=neural_mt_types.ENCODER_TYPE.new(
                rnn=neural_mt_types.RNN_ENCODER_PARAMS.new(
                    encoder_layer_configs=[
                        neural_mt_types.LAYER_CONFIG.new(
                            cell_type='lstm',
                            num_units=(
                                self.encoder_hidden_dim
                                if i > 0
                                else encoder_first_layer_dim
                            ),
                        ) for i in range(self.encoder_layers)
                    ],
                    use_bidirectional_encoder=self.encoder_bidirectional,
                ),
            ),
            decoder_type=neural_mt_types.DECODER_TYPE.new(
                rnn=neural_mt_types.RNN_DECODER_PARAMS.new(
                    decoder_layer_configs=[
                        neural_mt_types.LAYER_CONFIG.new(
                            cell_type='lstm',
                            num_units=self.decoder_embed_dim,
                        ),
                    ] * self.decoder_layers,
                ),
            ),
            batch_size=self.batch_size,
            optimizer=neural_mt_types.OPTIMIZER.new(
                gradient_descent=(
                    neural_mt_types.GRADIENT_DESCENT_OPTIMIZER_PARAMS.new(
                        learning_rate=self.lr[0],
                        learning_rate_decay_factor=self.lr_shrink,
                        use_locking=False,
                    )
                ),
            ),
            max_gradient_norm=self.clip_norm,
            decoder_softmax_size=self.decoder_out_embed_dim,
            label_smoothing_epsilon=self.label_smoothing_epsilon,
        )

        return partial_model_params


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_model_params_dict(partial_model_params=None):
    default_model_params = neural_mt_types.get_default_seq2seq_model_params()
    default_model_params_dict = {
        attr: getattr(default_model_params, attr)
        for attr in dir(default_model_params)
    }
    model_params_dict = dict(
        default_model_params_dict,
        **(
            partial_model_params
            if partial_model_params is not None
            else {}
        )
    )
    return model_params_dict


def build_caffe2_model(
    partial_model_params,
    source_vocab_size=103,
    target_vocab_size=103,
):
    '''vocab_size includes 100 reserved indices for special tokens.'''
    caffe2_model_params = neural_mt_types.SEQ2SEQ_MODEL_PARAMS.new(
        **get_model_params_dict(partial_model_params),
    )

    model_obj = seq2seq_model_caffe2.Seq2SeqModelCaffe2(
        model_params=caffe2_model_params,
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        num_gpus=0,
        num_samples=None,
        rnn_lm_model=None,
    )
    model_obj.initialize_from_scratch(
        initial_encoder_embeddings_model=None,
        initial_decoder_embeddings_model=None,
        initial_output_projection_embeddings_model=None,
        training_dir=None,
        source_vocab_processor=None,
        target_vocab_processor=None,
    )
    return model_obj


def dummy_dictionary(
    dummy_tokens=3,
    additional_token_list=None,
):
    """First adds the amount of dummy_tokens that you specify, then
    finally the additional_token_list, which is a list of string token values"""
    d = fbtranslate_dictionary.Dictionary()
    for i in range(dummy_tokens):
        token = 'token_{}'.format(i)
        d.add_symbol(token)
    if additional_token_list is not None:
        for token in additional_token_list:
            d.add_symbol(token)
    d.finalize()
    return d


def prepare_inputs(
    test_args,
    source_vocab_size=103,
    target_vocab_size=103,
    is_variable_seqlen=False,
):
    # first 100 indices are reserved for special tokens
    src_dict = dummy_dictionary(dummy_tokens=source_vocab_size - 100)
    tgt_dict = dummy_dictionary(dummy_tokens=source_vocab_size - 100)

    def get_single_example(
        sample_id,
        src_sentence_length,
        tgt_sentence_length,
    ):
        non_special_start = 4
        example = {
            'id': sample_id,
            # Note: both source and target-side sentences are expected
            # to end in the EOS marker. LanguagePairDataset then:
            # (1) moves the EOS to the start of the target, for input feeding
            # (2) it also handles left (right) padding of the source (target)
            'source': torch.LongTensor(
                np.random.randint(
                    low=non_special_start,
                    high=len(src_dict.symbols),
                    size=src_sentence_length,
                ).tolist()
                + [src_dict.eos()],
            ),

            'target': torch.LongTensor(
                np.random.randint(
                    low=non_special_start,
                    high=len(tgt_dict.symbols),
                    size=tgt_sentence_length,
                ).tolist()
                + [tgt_dict.eos()],
            ),
        }
        return example

    min_sent_len = 7
    max_sent_len = 12
    fixed_tgt_length = 12
    dataset = TestDataset([
        get_single_example(
            example_id,
            np.random.randint(
                low=min_sent_len,
                high=max_sent_len,
                size=1,
            ) if is_variable_seqlen else 10,
            fixed_tgt_length,
        ) for example_id in range(test_args.batch_size)
    ])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=test_args.batch_size,
        collate_fn=(
            lambda samples: data.LanguagePairDataset.collate(
                samples,
                src_dict.pad(),
                src_dict.eos(),
            )
        ),
    )
    data_iterator = iter(dataloader)
    return data_iterator, src_dict, tgt_dict


def lstm_reorder(pytorch_param):
    """
    PyTorch and Caffe2 both use combined weight matrices to represent the
    weights for each of the LSTM gates, but have different conventions for
    the order of those gates. This function converts from PyTorch convention
    to Caffe2 convention (or vice versa). Specifically the third and fourth
    segments need to be switched. Applies to weight matrix or bias vector.

    Pytorch: Input, Forget, Candidate, Output (IFGO)
    Caffe2:  Forget, Input, Output, Candidate (IFOG)
    """
    assert 1 <= (len(pytorch_param.shape)) <= 2
    total_gates_dim = pytorch_param.shape[0]
    assert total_gates_dim % 4 == 0
    gate_dim = total_gates_dim // 4

    result = pytorch_param.copy()
    if len(pytorch_param.shape) == 2:
        result[2 * gate_dim: 3 * gate_dim, :] = pytorch_param[3 * gate_dim:, :]
        result[3 * gate_dim:, :] = pytorch_param[2 * gate_dim: 3 * gate_dim, :]
    else:
        result[2 * gate_dim: 3 * gate_dim] = pytorch_param[3 * gate_dim:]
        result[3 * gate_dim:] = pytorch_param[2 * gate_dim: 3 * gate_dim]
    return result


def convert_encoder_cell_params_to_caffe2(pt_params, model_args):
    c2_params = {}

    for i in range(model_args.encoder_layers):
        if i == 0 and model_args.encoder_bidirectional:
            for direction in ('fw', 'bw'):
                c2_prefix = 'encoder/layer0/{}/lstm/'.format(direction)
                pt_prefix = 'encoder.layers.0.{}d_cell.'.format(direction)
                c2_params[c2_prefix + 'i2h_w'] = lstm_reorder(
                    pt_params[pt_prefix + 'weight_ih'],
                )
                c2_params[c2_prefix + 'i2h_b'] = lstm_reorder(
                    pt_params[pt_prefix + 'bias_ih'],
                )
                c2_params[c2_prefix + 'gates_t_w'] = lstm_reorder(
                    pt_params[pt_prefix + 'weight_hh'],
                )
                c2_params[c2_prefix + 'gates_t_b'] = lstm_reorder(
                    pt_params[pt_prefix + 'bias_hh'],
                )
            continue

        c2_params['encoder/layer{}/lstm/i2h_w'.format(i)] = lstm_reorder(
            pt_params['encoder.layers.{}.fwd_cell.weight_ih'.format(i)],
        )
        c2_params['encoder/layer{}/lstm/i2h_b'.format(i)] = lstm_reorder(
            pt_params['encoder.layers.{}.fwd_cell.bias_ih'.format(i)],
        )
        c2_params['encoder/layer{}/lstm/gates_t_w'.format(i)] = lstm_reorder(
            pt_params['encoder.layers.{}.fwd_cell.weight_hh'.format(i)],
        )
        c2_params['encoder/layer{}/lstm/gates_t_b'.format(i)] = lstm_reorder(
            pt_params['encoder.layers.{}.fwd_cell.bias_hh'.format(i)],
        )

    return c2_params


def convert_encoder_sequence_params_to_caffe2(pt_params, model_args):
    c2_params = {}

    for i in range(model_args.encoder_layers):
        if i == 0 and model_args.encoder_bidirectional:
            for direction in ('fw', 'bw'):
                c2_prefix = 'encoder/layer0/{}/lstm/'.format(direction)
                pt_prefix = 'encoder.layers.0.'
                pt_suffix = '_l0' if direction == 'fw' else '_l0_reverse'
                c2_params[c2_prefix + 'i2h_w'] = lstm_reorder(
                    pt_params[pt_prefix + 'weight_ih' + pt_suffix],
                )
                c2_params[c2_prefix + 'i2h_b'] = lstm_reorder(
                    pt_params[pt_prefix + 'bias_ih' + pt_suffix],
                )
                c2_params[c2_prefix + 'gates_t_w'] = lstm_reorder(
                    pt_params[pt_prefix + 'weight_hh' + pt_suffix],
                )
                c2_params[c2_prefix + 'gates_t_b'] = lstm_reorder(
                    pt_params[pt_prefix + 'bias_hh' + pt_suffix],
                )
            continue

        c2_params['encoder/layer{}/lstm/i2h_w'.format(i)] = lstm_reorder(
            pt_params['encoder.layers.{}.weight_ih_l0'.format(i)],
        )
        c2_params['encoder/layer{}/lstm/i2h_b'.format(i)] = lstm_reorder(
            pt_params['encoder.layers.{}.bias_ih_l0'.format(i)],
        )
        c2_params['encoder/layer{}/lstm/gates_t_w'.format(i)] = lstm_reorder(
            pt_params['encoder.layers.{}.weight_hh_l0'.format(i)],
        )
        c2_params['encoder/layer{}/lstm/gates_t_b'.format(i)] = lstm_reorder(
            pt_params['encoder.layers.{}.bias_hh_l0'.format(i)],
        )

    return c2_params


def convert_params_to_caffe2(model, model_args):
    pt_params = {}
    for name, param in model.named_parameters():
        pt_params[name] = param.data.numpy()

    c2_params = {}
    c2_params['encoder_embeddings'] = pt_params['encoder.embed_tokens.weight']
    c2_params['decoder_embeddings'] = pt_params['decoder.embed_tokens.weight']

    if model_args.sequence_lstm:
        c2_params.update(
            convert_encoder_sequence_params_to_caffe2(pt_params, model_args),
        )
    else:
        c2_params.update(
            convert_encoder_cell_params_to_caffe2(pt_params, model_args),
        )

    for i in range(model_args.decoder_layers):
        prefix = 'decoder/layer_{}/decoder_cell'.format(i)
        if i == 0:
            c2_params['decoder/decoder_cell/i2h_w'] = lstm_reorder(
                pt_params['decoder.layers.0.weight_ih'.format(i)],
            )
            c2_params['decoder/decoder_cell/i2h_b'] = lstm_reorder(
                pt_params['decoder.layers.0.bias_ih'.format(i)],
            )
        else:
            c2_params['{}/i2h_w'.format(prefix)] = lstm_reorder(
                pt_params['decoder.layers.{}.weight_ih'.format(i)],
            )
            c2_params['{}/i2h_b'.format(prefix)] = lstm_reorder(
                pt_params['decoder.layers.{}.bias_ih'.format(i)],
            )

        c2_params['{}/gates_t_w'.format(prefix)] = lstm_reorder(
            pt_params['decoder.layers.{}.weight_hh'.format(i)],
        )
        c2_params['{}/gates_t_b'.format(prefix)] = lstm_reorder(
            pt_params['decoder.layers.{}.bias_hh'.format(i)],
        )

        # In Caffe2, recurrent input-feeding is added to "gates", not input
        if i == 0:
            enc_dim = model_args.encoder_hidden_dim
            c2_params['{}/gates_t_w'.format(prefix)] = np.concatenate(
                [
                    c2_params['{}/gates_t_w'.format(prefix)],
                    c2_params['decoder/decoder_cell/i2h_w'][:, -enc_dim:],
                ],
                axis=1,
            )
            c2_params['decoder/decoder_cell/i2h_w'] = (
                c2_params['decoder/decoder_cell/i2h_w'][:, :-enc_dim]
            )

    c2_params['initial_attention_weighted_encoder_context'] = (
        pt_params['decoder.initial_attn_context']
    )

    if model_args.decoder_out_embed_dim != model_args.decoder_embed_dim:
        c2_params['decoder_outputs_scaled_w'] = (
            pt_params['decoder.additional_fc.weight']
        )
        c2_params['decoder_outputs_scaled_b'] = (
            pt_params['decoder.additional_fc.bias']
        )

    c2_params['output_projection_w'] = pt_params['decoder.output_projection_w']
    c2_params['output_projection_b'] = pt_params['decoder.output_projection_b']

    return c2_params


def load_caffe2_model(caffe2_params):
    for name, value in caffe2_params.items():
        old_value = workspace.FetchBlob(name)
        try:
            assert old_value.shape == value.shape
        except AssertionError:
            err = '{}: could not load shape {}. expected shape {}.'.format(
                name,
                value.shape,
                old_value.shape,
            )
            raise RuntimeError(err)
        workspace.FeedBlob(name, value)


def convert_batch_to_caffe2(sample):
    # source sample from fairseq DataLoader is left padded, which needs
    # to be converted to right padded whenever needed
    PAD_ID = 1
    encoder_inputs = sample['net_input']['src_tokens'].numpy().T
    encoder_lengths = (encoder_inputs != PAD_ID).sum(axis=0)
    if data.LanguagePairDataset.LEFT_PAD_SOURCE:
        encoder_inputs = utils.convert_padding_direction(
            sample['net_input']['src_tokens'],
            encoder_lengths,
            PAD_ID,
            left_to_right=True,
        ).numpy().T

    decoder_inputs = sample['net_input']['prev_output_tokens'].numpy().T
    targets = sample['target'].numpy().T
    decoder_lengths = (decoder_inputs != PAD_ID).sum(axis=0)
    target_weights = (targets != PAD_ID).astype('float32')

    return {
        'encoder_inputs': encoder_inputs.astype('int32'),
        'encoder_lengths': encoder_lengths.astype('int32'),
        'decoder_inputs': decoder_inputs.astype('int32'),
        'decoder_lengths': decoder_lengths.astype('int32'),
        'targets': targets.astype('int32'),
        'target_weights': target_weights,
    }


def caffe2_training_step(caffe2_model, caffe2_batch):
    for name, value in caffe2_batch.items():
        workspace.FeedBlob(name, value)
    workspace.FeedBlob(
        'total_target_tokens',
        np.array(caffe2_batch['target_weights'].sum()).astype('float32'),
    )
    workspace.RunNet(caffe2_model.model.net.Proto().name)
    return caffe2_model.total_loss_scalar()


def extract_caffe2_params(caffe2_model):
    caffe2_params = {}
    for param in caffe2_model.model.GetParams():
        value = workspace.FetchBlob(param)
        caffe2_params[str(param)] = value
    return caffe2_params

def create_caffe2_like_vocab_files():
    additional_special_tokens = CAFFE2_MAX_SPECIAL_TOKENS - \
        fbtranslate_dictionary.Dictionary().nspecial
    src_dict = dummy_dictionary(
        dummy_tokens=additional_special_tokens,
        additional_token_list=['a', 'b', 'c', 'd', 'e'],
    )
    tgt_dict = dummy_dictionary(
        dummy_tokens=additional_special_tokens,
        additional_token_list=['A', 'B', 'C', 'D', 'E'],
    )
    return src_dict, tgt_dict


def create_lexical_dictionaries():
    lexical_dictionary_path = write_lines_to_temp_file([
        'a A 0.7',
        'a B 0.3',
        'b C 0.1',
        'b D 0.8',
        'b E 0.1',
        'c A 0.3',
        'c B 0.4',
        'c C 0.3',
        'd D 0.4',
        'd E 0.3',
        'd A 0.2',
        'd B 0.1',
        'e C 1.0',
    ])
    return [lexical_dictionary_path]


def write_lines_to_temp_file(lines):
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        delete=False,
        dir='/tmp'
    )
    temp_file_path = temp_file.name
    temp_file.close()
    with codecs.open(temp_file_path, 'w', 'utf-8') as temp_file:
        temp_file.write('\n'.join(lines) + '\n')
    return temp_file_path


def compare_weights(caffe2_model, pytorch_model, test_args):
    caffe2_weights = extract_caffe2_params(caffe2_model)
    pytorch_weights = convert_params_to_caffe2(pytorch_model, test_args)
    for name in caffe2_weights:
        assert name in pytorch_weights
        is_weights_eq = np.allclose(
            caffe2_weights[name],
            pytorch_weights[name],
            atol=1e-4,
            rtol=1e-3,
        )
        if not is_weights_eq:
            for i in range(len(pytorch_weights[name])):
                if not np.allclose(
                    pytorch_weights[name][i],
                    caffe2_weights[name][i],
                    atol=1e-4,
                    rtol=1e-3,
                ):
                    print('index: {}'.format(i))
                    if isinstance(pytorch_weights[name][i], np.float32):
                        print(
                            '{}[{}]: {} vs. {}'.format(
                                name,
                                i,
                                pytorch_weights[name][i],
                                caffe2_weights[name][i]
                            )
                        )
                    else:
                        for j in range(len(pytorch_weights[name][i])):
                            if not np.allclose(
                                pytorch_weights[name][i][j],
                                caffe2_weights[name][i][j],
                                atol=1e-4,
                                rtol=1e-3,
                            ):
                                print('{}[{}][{}]: {} vs. {}'.format(
                                    name,
                                    i,
                                    j,
                                    pytorch_weights[name][i][j],
                                    caffe2_weights[name][i][j]
                                ))
        assert is_weights_eq
