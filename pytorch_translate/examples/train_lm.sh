#!/bin/bash

NCCL_ROOT_DIR="$(pwd)/nccl_2.1.15-1+cuda8.0_x86_64"
export NCCL_ROOT_DIR
LD_LIBRARY_PATH="${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar -xvzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
rm -rf checkpoints 1-billion-word-language-modeling-benchmark-r13output.tar.gz && mkdir -p checkpoints
cat 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00*-of-00100 > 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled.news.en
CUDA_VISIBLE_DEVICES=0 python3 pytorch_translate/train.py \
  "" \
  --log-verbose \
  --arch rnn \
  --cell-type lstm \
  --sequence-lstm \
  --max-tokens 999999 \
  --max-epoch 2 \
  --optimizer sgd \
  --lr 0.5 \
  --lr-shrink 0.95 \
  --clip-norm 5.0 \
  --encoder-dropout-in 0.1 \
  --encoder-dropout-out 0.1 \
  --decoder-dropout-in 0.2 \
  --decoder-dropout-out 0.2 \
  --criterion "label_smoothed_cross_entropy" \
  --label-smoothing 0.1 \
  --batch-size 64 \
  --encoder-bidirectional \
  --encoder-layers 2 \
  --encoder-embed-dim 256 \
  --encoder-hidden-dim 0 \
  --decoder-layers 2 \
  --decoder-embed-dim 256 \
  --decoder-hidden-dim 512 \
  --decoder-out-embed-dim 256 \
  --save-dir checkpoints \
  --attention-type no \
  --sentence-avg \
  --momentum 0 \
  --generate-bleu-eval-avg-checkpoint 10 \
  --beam 6 --no-beamable-mm --length-penalty 1.0 \
  --max-sentences 64 --max-sentences-valid 64 \
  --source-lang en \
  --target-lang en \
  --train-source-text-file 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled.news.en \
  --train-target-text-file 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled.news.en \
  --eval-source-text-file 1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100 \
  --eval-target-text-file 1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100 \
  --source-max-vocab-size 50000 \
  --target-max-vocab-size 50000 \
  --log-interval 500 \
  --seed "${RANDOM}" \
  2>&1 | tee -a checkpoints/log
