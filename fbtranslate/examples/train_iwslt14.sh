#!/bin/bash

. ~/miniconda/bin/activate
wget https://download.pytorch.org/models/translate/iwslt14/data.tar.gz
tar -xvzf data.tar.gz
rm -rf checkpoints && mkdir -p checkpoints
python fbtranslate/multiprocessing_train.py \
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
   --batch-size 64 \
   --lenpen 0 \
   --unkpen 0.5 \
   --word-reward 0.25 \
   --max-tokens 9999999 \
   --encoder-layers 2 \
   --encoder-embed-dim 256 \
   --encoder-hidden-dim 512 \
   --decoder-layers 2 \
   --decoder-embed-dim 256 \
   --decoder-hidden-dim 256 \
   --decoder-out-embed-dim 256 \
   --save-dir checkpoints \
   --attention-type dot \
   --sentence-avg \
   --momentum 0 \
   --generate-bleu-eval-avg-checkpoints 10 \
   --generate-bleu-eval-per-epoch \
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
   --log-interval 5000 \
   --seed "${RANDOM}"
