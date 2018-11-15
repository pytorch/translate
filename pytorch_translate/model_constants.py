#!/usr/bin/env python3

# Pretrained model params
PRETRAINED_CHAR_EMBED_DIM = 16
PRETRAINED_CHAR_CNN_PARAMS = [
    (32, 1),
    (32, 2),
    (64, 3),
    (128, 4),
    (256, 5),
    (512, 6),
    (1024, 7),
]
PRETRAINED_NUM_HIGHWAY_LAYERS = 2
PRETRAINED_CHAR_CNN_NONLINEAR_FN = "relu"
PRETRAINED_CHAR_CNN_OUTPUT_DIM = 512
