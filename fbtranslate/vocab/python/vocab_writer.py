#!/usr/bin/env python3

import language_technology.neural_mt.fbtranslate.vocab.\
    vocab_constants as constants
import tempfile


def write_tokens_to_file(
    vocab_file,
    tokens_and_counts,
    delim='\t',  # delim must be whitespace for Loader to read token list
):
    vocab_file.write(str(len(tokens_and_counts)) + '\n')
    for token_pair in tokens_and_counts:
        vocab_file.write('{}{}{}\n'.format(token_pair[0], delim, token_pair[1]))


def save_vocab_to_file(
    vocab_type,
    tokens_and_counts,
    type_specific_params_dict=None,
    vocab_file_path=None,
):
    '''
    Inputs:
        vocab_type: string
        tokens_and_counts: list of (string token, int count) tuples
        type_specific_params_dict: extra params used only for some vocab types
        vocab_file_path: path to write vocab to
    Returns:
        output_path: file path of the saved vocab file, if input
            vocab_file_path is None, this will be a temp file
    '''
    with (
        open(vocab_file_path, 'w')
        if vocab_file_path is not None
        else tempfile.NamedTemporaryFile(mode='w', delete=False)
    ) as vocab_file:
        # TODO: in a later diff, assert vocab type is in the list of supported
        # types in vocab_constants
        vocab_file.write(vocab_type + '\n')
        '''
        The vocab file will look like this:

        vocab_type
        40
        first 20
        second 10
        third 2
        ...

        '''
        write_tokens_to_file(vocab_file, tokens_and_counts)
        if vocab_type == constants.CHAR_NGRAM_VOCAB_TYPE:
            '''
            In addition to the token list, we append a list of ngrams:

            ngram_size 2
            100
            first ngram 20
            second ngram 10
            third ngram 2
            ...

            '''
            vocab_file.write('ngram_size {}\n'.format(
                type_specific_params_dict['ngram_size'],
            ))
            write_tokens_to_file(
                vocab_file,
                type_specific_params_dict['ngrams_and_counts'],
            )
        if vocab_type == constants.MORFESSOR_VOCAB_TYPE:
            '''
            In addition to the token list, we append a token count.
            '''
            vocab_file.write('token_count {}\n'.format(
                type_specific_params_dict['token_count'],
            ))
        output_path = vocab_file.name
    return output_path
