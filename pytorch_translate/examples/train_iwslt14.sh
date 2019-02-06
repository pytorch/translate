#!/bin/bash

NCCL_ROOT_DIR="$(pwd)/nccl_2.1.15-1+cuda8.0_x86_64"
export NCCL_ROOT_DIR
LD_LIBRARY_PATH="${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
wget https://download.pytorch.org/models/translate/iwslt14/data.tar.gz
tar -xvzf data.tar.gz
rm -rf checkpoints data.tar.gz && mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=0 python3 pytorch_translate/train.py \
   "" \
   --arch rnn \
   --log-verbose \
   --lr-scheduler fixed \
   --force-anneal 200 \
   --cell-type lstm \
   --sequence-lstm \
   --reverse-source \
   --encoder-bidirectional \
   --max-epoch 100 \
   --stop-time-hr 72 \
   --stop-no-best-bleu-eval 5 \
   --optimizer sgd \
   --lr 0.5 \
   --lr-shrink 0.95 \
   --clip-norm 5.0 \
   --encoder-dropout-in 0.1 \
   --encoder-dropout-out 0.1 \
   --decoder-dropout-in 0.2 \
   --decoder-dropout-out 0.2 \
   --criterion label_smoothed_cross_entropy \
   --label-smoothing 0.1 \
   --batch-size 256 \
   --length-penalty 0 \
   --unk-reward -0.5 \
   --word-reward 0.25 \
   --max-tokens 9999999 \
   --encoder-layers 2 \
   --encoder-embed-dim 256 \
   --encoder-hidden-dim 512 \
   --decoder-layers 2 \
   --decoder-embed-dim 256 \
   --decoder-hidden-dim 512 \
   --decoder-out-embed-dim 256 \
   --save-dir checkpoints \
   --attention-type dot \
   --sentence-avg \
   --momentum 0 \
   --num-avg-checkpoints 10 \
   --beam 6 \
   --no-beamable-mm \
   --source-lang de \
   --target-lang en \
   --train-source-text-file data/train.tok.bpe.de \
   --train-target-text-file data/train.tok.bpe.en \
   --eval-source-text-file data/valid.tok.bpe.de \
   --eval-target-text-file data/valid.tok.bpe.en \
   --source-max-vocab-size 14000 \
   --target-max-vocab-size 14000 \
   --log-interval 10 \
   --seed "${RANDOM}" \
   2>&1 | tee -a checkpoints/log
