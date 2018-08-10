#!/usr/bin/env python3

import sys
import argparse
import random


def load_dictionary(filename):
    """Loads a pytorch_translate dictionary"""
    dic = set()
    with open(filename, "r") as df:
        for line in df:
            # The format is [word] [frequency]
            # So discard the frequency part
            word = line.strip().split()[0]
            dic.add(word)
    return dic


def get_parser():
    parser = argparse.ArgumentParser("Convert <unk> to actual typos")
    parser.add_argument("SRC", type=str, metavar="SRC", help="Source file.")
    parser.add_argument(
        "ADV",
        type=str,
        metavar="ADV",
        help="Adversarial input."
    )
    parser.add_argument(
        "--dictionary-file",
        type=str,
        help="File containing the dictionary, one line per word.",
    )
    parser.add_argument(
        "--unk-token",
        type=str,
        default="<unk>",
        help="Token for unknown words."
    )
    parser.add_argument(
        "--num-scrambling",
        type=int,
        default="10",
        help="Number of times we'll try to swap characters to make an OOV."
    )
    return parser


def make_oov(word, dic, args):
    """Modify a word to make it OOV (while keeping the meaning)"""
    # If the word has more than 3 letters try scrambling them
    if len(word) > 3:
        # For a fixed number of steps
        for _ in range(args.num_scrambling):
            # Swap two adjacent characters at the middle of the word
            pos = random.randint(1, len(word)-3)
            word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
            # If we got an OOV already just return it
            if word not in dic:
                return word
    # If nothing worked, or the word is too short for scrambling, just repeat
    # the last character ad nauseam
    char = word[-1]
    while word in dic:
        word = word + char
    return word


def main():
    parser = get_parser()
    args = parser.parse_args()
    # Load the dictionary
    dic = load_dictionary(args.dictionary_file)
    # Process files
    with open(args.SRC, "r") as src_file, open(args.ADV, "r") as adv_file:
        for src_line, adv_line in zip(src_file, adv_file):
            # Read the original and adversarial source swords
            src_words = src_line.strip().split()
            adv_words = adv_line.strip().split()
            # This will be the new adversarial source
            new_adv_words = []
            for src_word, adv_word in zip(src_words, adv_words):
                # Check whether the current adversarial word is an unk
                # (if not then it is the same as the original input)
                if adv_word == args.unk_token:
                    if src_word in dic:
                        # If the word is in the dictionary this was probably
                        # introduced by the adversarial attack, modify the word
                        # to get an unk
                        new_adv_words.append(make_oov(src_word, dic, args))
                    else:
                        # Otherwise the unk was probably introduced by
                        # converting to and from integers, just replace it with
                        # the original word
                        new_adv_words.append(src_word)
                else:
                    new_adv_words.append(src_word)
            # Print the result to stdout
            print(" ".join(new_adv_words))



if __name__ == '__main__':
    main()
