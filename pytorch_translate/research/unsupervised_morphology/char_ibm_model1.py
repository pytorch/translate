#!/usr/bin/env python3

from pytorch_translate.research.unsupervised_morphology import ibm_model1


class CharIBMModel1(ibm_model1.IBMModel1):
    def __init__(self):
        super().__init__()
        self.eow_symbol = "_EOW"  # End of word symbol.
