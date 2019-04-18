#!/usr/bin/env python3

from pytorch_translate.research.unsupervised_morphology.bpe import BPE
from pytorch_translate.research.unsupervised_morphology.char_ibm_model1 import (
    CharIBMModel1,
)


class BilingualBPE(object):
    """
    An extension of the BPE model that is cross-lingual wrt parallel data.
    """

    def __init__(self):
        self.src_bpe = BPE()
        self.dst_bpe = BPE()
        self.src2dst_ibm_model = CharIBMModel1()
        self.dst2src_ibm_model = CharIBMModel1()

    def _init_params(
        self, src_txt_path: str, dst_txt_path: str, num_ibm_iters: int, num_cpus: int
    ):
        """
        Args:
            src_txt_path: Text path for source language in parallel data.
            dst_txt_path: Text path for target language in parallel data.
            num_ibm_iters: Number of training epochs for the IBM model.
            num_cpus: Number of CPUs for training the IBM model with multi-processing.
        """
        self.src_bpe._init_vocab(txt_path=src_txt_path)
        self.dst_bpe._init_vocab(txt_path=dst_txt_path)
        self.src2dst_ibm_model.learn_ibm_parameters(
            src_path=src_txt_path,
            dst_path=dst_txt_path,
            num_iters=num_ibm_iters,
            num_cpus=num_cpus,
        )
        self.dst2src_ibm_model.learn_ibm_parameters(
            src_path=dst_txt_path,
            dst_path=src_txt_path,
            num_iters=num_ibm_iters,
            num_cpus=num_cpus,
        )
