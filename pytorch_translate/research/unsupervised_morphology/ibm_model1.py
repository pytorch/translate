#!/usr/bin/env python3

from collections import defaultdict


class IBMModel1(object):
    def __init__(self):
        """
        translation_prob is the translation probability in the IBM model 1.
        the full pseudo-code is available at https://fburl.com/yvp31kuw
        """
        self.translation_prob = defaultdict()
        self.null_str = "<null>"

    def initialize_translation_probs(self, src_path, dst_path):
        """
        Direction of translation is conditioned on the source text: t(dst|src).
        """
        with open(src_path) as src_file, open(dst_path) as dst_file:
            for src_line, dst_line in zip(src_file, dst_file):
                src_words = set(src_line.strip().split() + [self.null_str])
                dst_words = set(dst_line.strip().split())
                for src_word in src_words:
                    if src_word not in self.translation_prob:
                        self.translation_prob[src_word] = defaultdict(float)
                    for dst_word in dst_words:
                        self.translation_prob[src_word][dst_word] = 1.0

        for src_word in self.translation_prob.keys():
            denom = len(self.translation_prob[src_word])
            for dst_word in self.translation_prob[src_word].keys():
                self.translation_prob[src_word][dst_word] = 1.0 / denom
