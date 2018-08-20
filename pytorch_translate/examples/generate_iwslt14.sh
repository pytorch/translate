#!/bin/bash

NCCL_ROOT_DIR="$(pwd)/nccl_2.1.15-1+cuda8.0_x86_64"
export NCCL_ROOT_DIR
LD_LIBRARY_PATH="${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
wget https://download.pytorch.org/models/translate/iwslt14/model.tar.gz https://download.pytorch.org/models/translate/iwslt14/data.tar.gz
tar -xvzf model.tar.gz
tar -xvzf data.tar.gz
rm -f data.tar.gz model.tar.gz

python3 pytorch_translate/generate.py \
       "" \
       --path model/averaged_checkpoint_best_0.pt \
       --source-vocab-file model/dictionary-de.txt \
       --target-vocab-file model/dictionary-en.txt \
       --source-text-file data/test.tok.bpe.de \
       --target-text-file data/test.tok.bpe.en \
       --unk-reward -0.5 \
       --length-penalty 0 \
       --word-reward 0.25 \
       --beam 6 \
       --remove-bpe \
       --quiet

# output should look like:
# | Translated 6750 sentences (152251 tokens) in 37.9s (4018.00 tokens/s)
# | Generate test with beam=6: BLEU4 = 31.31, 65.7/39.2/25.2/16.6 (BP=0.971, ratio=0.972, syslen=127453, reflen=131152)

python3 pytorch_translate/generate.py \
       "" \
       --path model/averaged_checkpoint_best_0.pt:model/averaged_checkpoint_best_1.pt \
       --source-vocab-file model/dictionary-de.txt \
       --target-vocab-file model/dictionary-en.txt \
       --source-text-file data/test.tok.bpe.de \
       --target-text-file data/test.tok.bpe.en \
       --unk-reward -0.5 \
       --length-penalty 0 \
       --word-reward 0.25 \
       --beam 6 \
       --remove-bpe \
       --quiet

# output should look like:
# | Translated 6750 sentences (152251 tokens) in 60.2s (2530.11 tokens/s)
# | Generate test with beam=6: BLEU4 = 32.88, 67.4/41.2/27.1/18.2 (BP=0.962, ratio=0.962, syslen=126199, reflen=131152)
# Notice how the performance improves when using two checkpoints instead of one: ensembling models is a very useful
# technique to improve translation quality.
