#!/usr/bin/env python3

import argparse
import os
import tempfile
from typing import List, Optional

from pytorch_translate import constants, options as pytorch_translate_options
from pytorch_translate.data import char_data, data as pytorch_translate_data
from pytorch_translate.dictionary import Dictionary


def validate_args(args):
    if not (
        (
            args.train_source_text_file
            or args.train_source_binary_path
            or args.multiling_train_source_text_file
        )
        and (
            args.train_target_text_file
            or args.train_target_binary_path
            or args.multiling_train_target_text_file
        )
        and (
            args.eval_source_text_file
            or args.eval_source_binary_path
            or args.multiling_eval_source_text_file
        )
        and (
            args.eval_target_text_file
            or args.eval_target_binary_path
            or args.multiling_eval_target_text_file
        )
    ):
        raise ValueError(
            "At least one of --*_text_file or --*_binary_path flags must be "
            "specified for each of --{train, eval}_{source, target}_*"
        )

    for file_type in (
        "train_source_text_file",
        "train_target_text_file",
        "eval_source_text_file",
        "eval_target_text_file",
    ):
        text_file = getattr(args, file_type)
        if text_file and not os.path.isfile(text_file):
            raise ValueError(
                f"Please specify an existing text file for "
                f"--{file_type}={text_file}"
            )

    for file_type in ("source_vocab_file", "target_vocab_file"):
        vocab_file = getattr(args, file_type)
        if not vocab_file:
            raise ValueError(
                f"--{file_type} must be specified - even if you don't have "
                f"a vocab file, you must still specify a location "
                f"for it to be written to."
            )


def maybe_generate_temp_file_path(output_path=None):
    """
    This function generates a temp file path if output_path is empty or None.
    This is useful to do before calling any preprocessing function that has a
    precondition that data path arguments are set to valid filepaths.
    """
    if not output_path:
        fd, output_path = tempfile.mkstemp()
        # We only need a unique file name since the helper functions
        # take care of actually creating the file.
        os.close(fd)
    # numpy silently appends this suffix if it is not present, so this ensures
    # that the correct path is returned
    if not output_path.endswith(".npz"):
        output_path += ".npz"
    return output_path


def binarize_text_file(
    text_file: str,
    dictionary: Dictionary,
    output_path: str,
    append_eos: bool,
    reverse_order: bool,
    use_char_data: bool = False,
    embed_bytes: bool = False,
    char_dictionary: Optional[Dictionary] = None,
    already_numberized: bool = False,
) -> str:
    output_path = maybe_generate_temp_file_path(output_path)
    if use_char_data:
        dataset = char_data.InMemoryNumpyWordCharDataset()
        dataset.parse(
            path=text_file,
            word_dict=dictionary,
            char_dict=char_dictionary,
            embed_bytes=embed_bytes,
            reverse_order=reverse_order,
            append_eos=append_eos,
        )
    else:
        dataset = pytorch_translate_data.InMemoryNumpyDataset()
        dataset.parse(
            path=text_file,
            dictionary=dictionary,
            reverse_order=reverse_order,
            append_eos=append_eos,
            already_numberized=already_numberized,
        )
    dataset.save(output_path)
    return output_path


def make_multiling_corpus_configs(
    language_ids,
    text_files,
    dictionaries,
    char_dictionaries=None,
    oversampling_rates=None,
):
    if not oversampling_rates:
        oversampling_rates = [1] * len(language_ids)
    if char_dictionaries is None:
        char_dictionaries = [None] * len(language_ids)
    assert len(language_ids) == len(text_files)
    assert len(language_ids) == len(dictionaries)
    assert len(language_ids) == len(oversampling_rates)
    return [
        pytorch_translate_data.MultilingualCorpusConfig(
            dialect_id=None
            if i is None
            else i + pytorch_translate_data.MULTILING_DIALECT_ID_OFFSET,
            data_file=p,
            dict=d,
            char_dict=cd,
            oversampling=o,
        )
        for i, p, d, cd, o in zip(
            language_ids,
            text_files,
            dictionaries,
            char_dictionaries,
            oversampling_rates,
        )
    ]


def binarize_text_file_multilingual(
    corpus_configs: List[pytorch_translate_data.MultilingualCorpusConfig],
    output_path: str,
    append_eos: bool,
    reverse_order: bool,
    prepend_language_id: bool,
    use_char_data: bool = False,
    embed_bytes: bool = False,
    already_numberized: bool = False,
) -> str:
    output_path = maybe_generate_temp_file_path(output_path)
    if use_char_data:
        dataset = char_data.InMemoryNumpyWordCharDataset()
        dataset.parse_multilingual(
            corpus_configs,
            reverse_order=reverse_order,
            append_eos=append_eos,
            embed_bytes=embed_bytes,
            prepend_language_id=prepend_language_id,
            already_numberized=already_numberized,
        )
    else:
        dataset = pytorch_translate_data.InMemoryNumpyDataset()
        dataset.parse_multilingual(
            corpus_configs,
            append_eos=append_eos,
            reverse_order=reverse_order,
            prepend_language_id=prepend_language_id,
            already_numberized=already_numberized,
        )
    dataset.save(output_path)
    return output_path


def preprocess_corpora(args):
    args.train_source_binary_path = maybe_generate_temp_file_path(
        args.train_source_binary_path
    )
    args.train_target_binary_path = maybe_generate_temp_file_path(
        args.train_target_binary_path
    )
    args.eval_source_binary_path = maybe_generate_temp_file_path(
        args.eval_source_binary_path
    )
    args.eval_target_binary_path = maybe_generate_temp_file_path(
        args.eval_target_binary_path
    )

    # Additional text preprocessing options could be added here before
    # binarizing.
    if pytorch_translate_data.is_multilingual(args):
        preprocess_corpora_multilingual(args)
    else:

        # Vocabs are built before preprocessing because we might need to use
        # both monolingual and bilingual corpora sources to build the vocab
        # (in the case of semisupervised training)
        source_dict, char_source_dict, target_dict = build_vocabs(args=args)

        preprocess_bilingual_corpora(
            args=args,
            source_dict=source_dict,
            char_source_dict=char_source_dict,
            target_dict=target_dict,
        )
        # Binarize additional monolingual corpora for the semisupervised translation
        # task
        if (
            args.task == constants.SEMI_SUPERVISED_TASK
            or args.task == constants.DENOISING_AUTOENCODER_TASK
        ):
            args.train_mono_source_binary_path = maybe_generate_temp_file_path(
                output_path=getattr(args, "train_mono_source_binary_path", None)
            )
            args.train_mono_target_binary_path = maybe_generate_temp_file_path(
                output_path=getattr(args, "train_mono_target_binary_path", None)
            )
            preprocess_monolingual_corpora(
                args,
                source_dict=source_dict,
                char_source_dict=char_source_dict,
                target_dict=target_dict,
            )


def preprocess_monolingual_corpora(
    args: argparse.Namespace,
    source_dict: Dictionary,
    char_source_dict: Dictionary,
    target_dict: Dictionary,
):
    """
    Preprocess source and target monolingual datasets
    Prerequisite: Vocabs are already built (see build_vocabs)
    """
    use_char_source = char_source_dict is not None
    if getattr(args, "train_mono_source_text_file", None):
        args.train_mono_source_binary_path = binarize_text_file(
            text_file=args.train_mono_source_text_file,
            dictionary=source_dict,
            output_path=args.train_mono_source_binary_path,
            append_eos=args.append_eos_to_source,
            reverse_order=args.reverse_source,
            use_char_data=use_char_source,
            char_dictionary=char_source_dict,
        )

    # For target sentences, we always append EOS tokens, and never reverse
    # their order.
    if getattr(args, "train_mono_target_text_file", None):
        args.train_mono_target_binary_path = binarize_text_file(
            text_file=args.train_mono_target_text_file,
            dictionary=target_dict,
            output_path=args.train_mono_target_binary_path,
            # We always append EOS to the target sentence since we still want
            # the model to output an indication the sentence has finished, even
            # if we don't append the EOS symbol to the source sentence
            # (to prevent the model from misaligning UNKs or other words
            # to the frequently occurring EOS).
            append_eos=True,
            # We don't reverse the order of the target sentence, since
            # even if the source sentence is fed to the model backwards,
            # we still want the model to start outputting from the first word.
            reverse_order=False,
        )


def build_vocabs(args: argparse.Namespace):
    """
    Builds vocabs or loads them from existing vocab files. If args.task
    is pytorch_translate_semi_supervised, we use the monolingual corpora in
    addition to the parallel corpora for building source and target vocabs.
    """
    source_files = [args.train_source_text_file]
    target_files = [args.train_target_text_file]

    if args.task == constants.SEMI_SUPERVISED_TASK and getattr(
        args, "add_monolingual_data_for_vocab_building", None
    ):
        if getattr(args, "train_mono_source_text_file", None):
            source_files.append(args.train_mono_source_text_file)
        if getattr(args, "train_mono_target_text_file", None):
            target_files.append(args.train_mono_target_text_file)

    source_dict = Dictionary.build_vocab_file_if_nonexistent(
        corpus_files=source_files,
        vocab_file=args.source_vocab_file,
        max_vocab_size=args.source_max_vocab_size,
        tokens_with_penalty=None,
    )
    use_char_source = (args.char_source_vocab_file != "") or (
        getattr(args, "arch", "") == "char_source"
    )
    char_source_dict = None
    if use_char_source:
        embed_bytes = getattr(args, "embed_bytes", False)
        char_source_dict = Dictionary.build_vocab_file_if_nonexistent(
            corpus_files=source_files,
            vocab_file=args.char_source_vocab_file,
            max_vocab_size=args.char_source_max_vocab_size,
            tokens_with_penalty=None,
            is_char_vocab=True,
            embed_bytes=embed_bytes,
        )

    target_dict = Dictionary.build_vocab_file_if_nonexistent(
        corpus_files=target_files,
        vocab_file=args.target_vocab_file,
        max_vocab_size=args.target_max_vocab_size,
        tokens_with_penalty=args.penalized_target_tokens_file,
    )
    return source_dict, char_source_dict, target_dict


def preprocess_bilingual_corpora(
    args: argparse.Namespace,
    source_dict: Dictionary,
    char_source_dict: Dictionary,
    target_dict: Dictionary,
):
    """
    Preprocess source and target parallel datasets
    Prerequisite: Vocabs are already built (see build_vocabs)
    """
    use_char_source = args.char_source_vocab_file != ""
    embed_bytes = getattr(args, "embed_bytes", False)
    if args.train_source_text_file:
        args.train_source_binary_path = binarize_text_file(
            text_file=args.train_source_text_file,
            dictionary=source_dict,
            output_path=args.train_source_binary_path,
            append_eos=args.append_eos_to_source,
            reverse_order=args.reverse_source,
            use_char_data=use_char_source,
            embed_bytes=embed_bytes,
            char_dictionary=char_source_dict,
        )
    if args.eval_source_text_file:
        args.eval_source_binary_path = binarize_text_file(
            text_file=args.eval_source_text_file,
            dictionary=source_dict,
            output_path=args.eval_source_binary_path,
            append_eos=args.append_eos_to_source,
            reverse_order=args.reverse_source,
            use_char_data=use_char_source,
            embed_bytes=embed_bytes,
            char_dictionary=char_source_dict,
        )

    # For target sentences, we always append EOS tokens, and never reverse
    # their order.
    if args.train_target_text_file:
        args.train_target_binary_path = binarize_text_file(
            text_file=args.train_target_text_file,
            dictionary=target_dict,
            output_path=args.train_target_binary_path,
            # We always append EOS to the target sentence since we still want
            # the model to output an indication the sentence has finished, even
            # if we don't append the EOS symbol to the source sentence
            # (to prevent the model from misaligning UNKs or other words
            # to the frequently occurring EOS).
            append_eos=True,
            # We don't reverse the order of the target sentence, since
            # even if the source sentence is fed to the model backwards,
            # we still want the model to start outputting from the first word.
            reverse_order=False,
        )
    if args.eval_target_text_file:
        args.eval_target_binary_path = binarize_text_file(
            text_file=args.eval_target_text_file,
            dictionary=target_dict,
            output_path=args.eval_target_binary_path,
            append_eos=True,
            reverse_order=False,
        )


def build_vocab_multicorpus(
    corpus_langs,
    corpus_files,
    vocab_langs,
    vocab_files,
    max_vocab_size,
    tokens_with_penalty=None,
):
    lang2corpus = {lang: [] for lang in vocab_langs}
    for lang, corpus_file in zip(corpus_langs, corpus_files):
        lang2corpus[lang].append(corpus_file)
    return {
        lang: Dictionary.build_vocab_file_if_nonexistent(
            corpus_files=lang2corpus[lang],
            vocab_file=vocab_file,
            max_vocab_size=max_vocab_size,
            tokens_with_penalty=tokens_with_penalty,
        )
        for lang, vocab_file in zip(vocab_langs, vocab_files)
    }


def preprocess_corpora_multilingual(args):
    source_dicts = build_vocab_multicorpus(
        args.multiling_source_lang,
        args.multiling_train_source_text_file,
        args.multiling_encoder_lang,
        args.multiling_source_vocab_file,
        args.source_max_vocab_size,
    )
    source_corpus_lang_ids = [
        args.multiling_encoder_lang.index(l) for l in args.multiling_source_lang
    ]
    source_corpus_dicts = [source_dicts[l] for l in args.multiling_source_lang]
    binarize_text_file_multilingual(
        corpus_configs=make_multiling_corpus_configs(
            source_corpus_lang_ids,
            args.multiling_train_source_text_file,
            source_corpus_dicts,
            args.multiling_train_oversampling,
        ),
        output_path=args.train_source_binary_path,
        append_eos=args.append_eos_to_source,
        reverse_order=args.reverse_source,
        prepend_language_id=False,
    )
    binarize_text_file_multilingual(
        corpus_configs=make_multiling_corpus_configs(
            source_corpus_lang_ids,
            args.multiling_eval_source_text_file,
            source_corpus_dicts,
            args.multiling_train_oversampling,
        ),
        output_path=args.eval_source_binary_path,
        append_eos=args.append_eos_to_source,
        reverse_order=args.reverse_source,
        prepend_language_id=False,
    )

    target_dicts = build_vocab_multicorpus(
        args.multiling_target_lang,
        args.multiling_train_target_text_file,
        args.multiling_decoder_lang,
        args.multiling_target_vocab_file,
        args.target_max_vocab_size,
        args.penalized_target_tokens_file,
    )
    target_corpus_lang_ids = [
        args.multiling_decoder_lang.index(l) for l in args.multiling_target_lang
    ]
    target_corpus_dicts = [target_dicts[l] for l in args.multiling_target_lang]
    binarize_text_file_multilingual(
        corpus_configs=make_multiling_corpus_configs(
            target_corpus_lang_ids,
            args.multiling_train_target_text_file,
            target_corpus_dicts,
            args.multiling_train_oversampling,
        ),
        output_path=args.train_target_binary_path,
        append_eos=True,
        reverse_order=False,
        prepend_language_id=True,
    )
    binarize_text_file_multilingual(
        corpus_configs=make_multiling_corpus_configs(
            target_corpus_lang_ids,
            args.multiling_eval_target_text_file,
            target_corpus_dicts,
            args.multiling_train_oversampling,
        ),
        output_path=args.eval_target_binary_path,
        append_eos=True,
        reverse_order=False,
        prepend_language_id=True,
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Translate - preprocessing")
    pytorch_translate_options.add_verbosity_args(parser)
    pytorch_translate_options.add_preprocessing_args(parser)
    args = parser.parse_args()
    pytorch_translate_options.validate_preprocessing_args(args)
    pytorch_translate_options.print_args(args)
    preprocess_corpora(args)


if __name__ == "__main__":
    main()
