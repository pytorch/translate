#!/usr/bin/env python3

import codecs
import logging

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)

# Vocab reduction default params when only lexical dictionaries is provided
NUM_TOP_WORDS_DEFAULT = 2000

MAX_TRANSLATION_CANDIDATES_PER_WORD_DEFAULT = 30


def add_args(parser):
    parser.add_argument(
        "--lexical-dictionary",
        type=str,
        metavar="EXPR",
        action="append",
        help=("lexical dictionary(ies) for vocab reduction"),
    )
    parser.add_argument(
        "--num-top-words",
        type=int,
        metavar="N",
        help="num top words for vocab reduction",
    )
    parser.add_argument(
        "--max-translation-candidates-per-word",
        type=int,
        metavar="N",
        help="max translation candidates per word for vocab reduction",
    )


def set_arg_defaults(args):
    # lexical_dictionaries is the only required argument for vocab reduction
    lexical_dictionaries = getattr(args, "lexical_dictionary", None)
    if hasattr(args, "vocab_reduction_params"):
        # We've already created the vocab reduction params from the bottom-level
        # lexical_dictionaries, num_top_words and
        # max_translation_candidates_per_word args
        return args.vocab_reduction_params
    args.vocab_reduction_params = None
    if lexical_dictionaries is not None:
        num_top_words = getattr(args, "num_top_words", NUM_TOP_WORDS_DEFAULT)
        max_translation_candidates_per_word = getattr(
            args,
            "max_translation_candidates_per_word",
            MAX_TRANSLATION_CANDIDATES_PER_WORD_DEFAULT,
        )
        args.vocab_reduction_params = {
            "lexical_dictionaries": lexical_dictionaries,
            "num_top_words": num_top_words,
            "max_translation_candidates_per_word": max_translation_candidates_per_word,
        }
        # For less redundant logging when we print out the args Namespace,
        # delete the bottom-level args, since we'll just be dealing with
        # args.vocab_reduction_params from now on
        delattr(args, "lexical_dictionary")
        if hasattr(args, "num_top_words"):
            delattr(args, "num_top_words")
        if hasattr(args, "max_translation_candidates_per_word"):
            delattr(args, "max_translation_candidates_per_word")


def select_top_candidate_per_word(
    source_index,
    target_indices_with_prob,
    counter_per_word,
    max_translation_candidates_per_word,
    translation_candidates,
    translation_candidates_set,
):
    translation_candidates_saved = 0
    target_indices_with_prob.sort(key=lambda x: x[1], reverse=True)
    for target_index_with_prob in target_indices_with_prob:
        if counter_per_word[source_index] >= max_translation_candidates_per_word:
            # don't save more than max_translation_candidates_per_word
            # translation candidates for any one source token
            break

        # update translation candidates matrix at [source index, running counter
        # per source token] to = target index
        translation_candidates[
            source_index, counter_per_word[source_index]
        ] = target_index_with_prob[0]
        translation_candidates_set.update((source_index, target_index_with_prob[0]))
        counter_per_word[source_index] += 1
        translation_candidates_saved += 1
    return translation_candidates_saved


def get_translation_candidates(
    src_dict,
    dst_dict,
    lexical_dictionaries,
    num_top_words,
    max_translation_candidates_per_word,
):
    """
    Reads a lexical dictionary file, where each line is (source token, possible
    translation of source token, probability). The file is generally grouped
    by source tokens, but within the group, the probabilities are not
    necessarily sorted.

    A a 0.3
    A c 0.1
    A e 0.05
    A f 0.01
    B b 0.6
    B b 0.2
    A z 0.001
    A y 0.002
    ...

    Returns: translation_candidates
        Matrix of shape (src_dict, max_translation_candidates_per_word) where
        each row corresponds to a source word in the vocab and contains token
        indices of translation candidates for that source word
    """

    translation_candidates = np.zeros(
        [len(src_dict), max_translation_candidates_per_word], dtype=np.int32
    )

    # running count of translation candidates per source word
    counter_per_word = np.zeros(len(src_dict), dtype=np.int32)

    # tracks if we've already seen some (source token, target token) pair so we
    # ignore duplicate lines
    translation_candidates_set = set()

    for lexical_dictionary in lexical_dictionaries:
        logger.info(f"Processing dictionary file {lexical_dictionary}")
        translation_candidates_saved = 0

        with codecs.open(lexical_dictionary, "r", "utf-8") as lexical_dictionary_file:
            current_source_index = None
            current_target_indices = []
            for line in lexical_dictionary_file.readlines():
                alignment_data = line.split()
                if len(alignment_data) != 3:
                    logger.warning(f"Malformed line in lexical dictionary: {line}")
                    continue
                source_word, target_word, prob = alignment_data
                prob = float(prob)
                source_index = src_dict.index(source_word)
                target_index = dst_dict.index(target_word)
                if (
                    source_index not in src_dict.lexicon_indices
                    and target_index in dst_dict.lexicon_indices
                ):
                    continue

                if source_index is not None and target_index is not None:
                    if source_index != current_source_index:
                        # We've finished processing the possible translation
                        # candidates for this source token group, so save the
                        # extracted translation candidates
                        translation_candidates_saved += select_top_candidate_per_word(
                            current_source_index,
                            current_target_indices,
                            counter_per_word,
                            max_translation_candidates_per_word,
                            translation_candidates,
                            translation_candidates_set,
                        )
                        current_source_index = source_index
                        current_target_indices = []

                    if (
                        target_index >= num_top_words
                        and (source_index, target_index)
                        not in translation_candidates_set
                    ):
                        current_target_indices.append((target_index, prob))
        # Save the extracted translation candidates for the last source token
        # group
        translation_candidates_saved += select_top_candidate_per_word(
            current_source_index,
            current_target_indices,
            counter_per_word,
            max_translation_candidates_per_word,
            translation_candidates,
            translation_candidates_set,
        )
        logger.info(
            f"Loaded {translation_candidates_saved} translation"
            f"candidates from dictionary {lexical_dictionary}"
        )
    return translation_candidates


class VocabReduction(nn.Module):
    def __init__(
        self,
        src_dict,
        dst_dict,
        vocab_reduction_params,
        predictor=None,
        fp16: bool = False,
    ):
        super().__init__()
        self.src_dict = src_dict
        self.dst_dict = dst_dict
        self.vocab_reduction_params = vocab_reduction_params
        self.predictor = predictor
        self.fp16 = fp16
        self.translation_candidates = None

        if (
            self.vocab_reduction_params is not None
            and self.vocab_reduction_params["max_translation_candidates_per_word"] > 0
        ):
            translation_candidates = get_translation_candidates(
                self.src_dict,
                self.dst_dict,
                self.vocab_reduction_params["lexical_dictionaries"],
                self.vocab_reduction_params["num_top_words"],
                self.vocab_reduction_params["max_translation_candidates_per_word"],
            )
            self.translation_candidates = nn.Parameter(
                torch.Tensor(translation_candidates).long(), requires_grad=False
            )

    # encoder_output is default None for backwards compatibility
    def forward(self, src_tokens, encoder_output=None, decoder_input_tokens=None):
        assert self.dst_dict.pad() == 0, (
            f"VocabReduction only works correctly when the padding ID is 0 "
            "(to ensure its position in possible_translation_tokens is also 0), "
            f"instead of {self.dst_dict.pad()}."
        )
        vocab_list = [src_tokens.new_tensor([self.dst_dict.pad()])]

        if decoder_input_tokens is not None:
            flat_decoder_input_tokens = decoder_input_tokens.view(-1)
            vocab_list.append(flat_decoder_input_tokens)

        if self.translation_candidates is not None:
            reduced_vocab = self.translation_candidates.index_select(
                dim=0, index=src_tokens.view(-1)
            ).view(-1)
            vocab_list.append(reduced_vocab)
        if (
            self.vocab_reduction_params is not None
            and self.vocab_reduction_params["num_top_words"] > 0
        ):
            top_words = torch.arange(
                self.vocab_reduction_params["num_top_words"],
                device=vocab_list[0].device,
            ).long()
            vocab_list.append(top_words)

        # Get bag of words predicted by word predictor
        if self.predictor is not None:
            assert encoder_output is not None
            pred_output = self.predictor(encoder_output)
            # [batch, k]
            topk_indices = self.predictor.get_topk_predicted_tokens(
                pred_output, src_tokens, log_probs=True
            )
            # flatten indices for entire batch [1, batch * k]
            topk_indices = topk_indices.view(-1)
            vocab_list.append(topk_indices.detach())

        all_translation_tokens = torch.cat(vocab_list, dim=0)
        possible_translation_tokens = torch.unique(
            all_translation_tokens,
            # Sorting helps ensure that the padding ID (0) remains in position 0.
            sorted=True,
            # The decoder_input_tokens used here are very close to the targets
            # tokens that we also need to map to the reduced vocab space later
            # on, except that decoder_input_tokens have <eos> prepended, while
            # the targets will have <eos> at the end of the sentence. This
            # prevents us from being able to directly use the inverse indices
            # that torch.unique can return.
            return_inverse=False,
        ).type_as(src_tokens)

        # Pad to a multiple of 8 to ensure training with fp16 will activate
        # NVIDIA Tensor Cores.
        len_mod_eight = possible_translation_tokens.shape[0] % 8
        if self.training and self.fp16 and len_mod_eight != 0:
            possible_translation_tokens = torch.cat(
                [
                    possible_translation_tokens,
                    possible_translation_tokens.new_tensor(
                        [self.dst_dict.pad()] * (8 - len_mod_eight)
                    ),
                ]
            )

        return possible_translation_tokens
