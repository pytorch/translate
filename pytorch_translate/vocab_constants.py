#!/usr/bin/env python3

MAX_SPECIAL_TOKENS = 100

# Number of Byte indices is always fixed at 256 (0-255). The additional 5 indices
# correpsond to the special tokens for byte numberization including
# padding, start and end of word, start and end of sentence. These are
# separate from the special tokens in the dict and match up with the indices
# used by pre-trained ELMo.
NUM_BYTE_INDICES = 261

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4
