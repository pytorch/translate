#!/bin/bash

# Retrieve global variables
source scripts/globals.sh

# Experiment variables
SRC="fr"
TRG="en"
EXP_NAME="iwslt2016-${SRC}-${TRG}-transformer-base-40k"

# Checkpoint (where the models will be saved)
CKPT_DIR=${CHECKPOINTS}/${EXP_NAME}
mkdir -p $CKPT_DIR
echo "Using checkpoint dir: ${CKPT_DIR}"

# Data
DATA_DIR=${IWSLT2016_DIR}/${SRC}-${TRG}
DATA_BIN_DIR="${DATA_DIR}/bin"
mkdir -p $DATA_BIN_DIR

# Unique ID for temp files
TMP_ID=`uuidgen`


############
# Training #
############

# Training command
python $PTT_DIR/train.py \
  --log-verbose  \
  --args-verbosity 2 \
  --no-progress-bar  \
  --arch ptt_transformer \
  --max-tokens 4000 \
  --max-sentences-valid 10 \
  --max-epoch 60 \
  --stop-time-hr 70 \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 \
  --warmup-updates 4000 \
  --lr 0.0005 \
  --min-lr 1e-09 \
  --dropout 0.3 \
  --weight-decay 0.0001 \
  --criterion "label_smoothed_cross_entropy" \
  --label-smoothing 0.1 \
  --encoder-layers 6 \
  --encoder-embed-dim 512 \
  --encoder-ffn-embed-dim 1024 \
  --encoder-attention-heads 4 \
  --decoder-layers 6 \
  --decoder-embed-dim 512 \
  --decoder-ffn-embed-dim 1024 \
  --decoder-attention-heads 4 \
  --generate-bleu-eval-per-epoch  \
  --beam 5 \
  --length-penalty 1.0 \
  --source-max-vocab-size 40000 \
  --target-max-vocab-size 40000 \
  --share-decoder-input-output-embed True \
  --log-interval 500 \
  --seed 42 \
  --max-checkpoints-kept 10 \
  --generate-bleu-eval-avg-checkpoints 10 \
  --source-lang ${SRC} \
  --target-lang ${TRG} \
  --save-dir ${CKPT_DIR} \
  --train-source-text-file ${DATA_DIR}/train.${SRC}-${TRG}.tok.${SRC} \
  --train-target-text-file ${DATA_DIR}/train.${SRC}-${TRG}.tok.${TRG} \
  --eval-source-text-file ${DATA_DIR}/dev.${SRC}-${TRG}.tok.${SRC} \
  --eval-target-text-file ${DATA_DIR}/dev.${SRC}-${TRG}.tok.${TRG} \
  --train-source-binary-path ${DATA_BIN_DIR}/train.${SRC}-${TRG}.${SRC} \
  --train-target-binary-path ${DATA_BIN_DIR}/train.${SRC}-${TRG}.${TRG} \
  --eval-source-binary-path ${DATA_BIN_DIR}/dev.${SRC}-${TRG}.${SRC} \
  --eval-target-binary-path ${DATA_BIN_DIR}/dev.${SRC}-${TRG}.${TRG}


##############
# Evaluation #
##############


# Helper function to compute BLEU with uniform tokenization
function bleu (){
  REF_FILE=$1
  HYP_FILE=$2
  # Detokenize/de-bpe first just in case
  cat $REF_FILE | sed -r 's/(@@ )|(@@ ?$)//g' | \
    perl $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $TRG \
    > /tmp/gold.${TMP_ID}.txt
  cat $HYP_FILE | sed -r 's/(@@ )|(@@ ?$)//g' | \
    perl $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $TRG \
    > /tmp/hyp.${TMP_ID}.txt
  # Bleu
  perl $MOSES_SCRIPTS/generic/multi-bleu-detok.perl \
    /tmp/gold.${TMP_ID}.txt < /tmp/hyp.${TMP_ID}.txt
}

# Run on the dev set
python $PTT_DIR/generate.py \
  --no-progress-bar \
  --path $CKPT_DIR/averaged_checkpoint_best.pt \
  --source-vocab-file $CKPT_DIR/dictionary-${SRC}.txt \
  --target-vocab-file $CKPT_DIR/dictionary-${TRG}.txt \
  --source-text-file ${DATA_DIR}/dev.${SRC}-${TRG}.tok.${SRC} \
  --target-text-file ${DATA_DIR}/dev.${SRC}-${TRG}.tok.${TRG} \
  --beam 5 \
  --length-penalty 1.0 \
  --max-tokens 2000 \
  --quiet \
  --translation-output-file $CKPT_DIR/dev.${SRC}-${TRG}.tok.out.${TRG}

# Compute dev BLEU score
echo "Detokenized BLEU score on the dev set"
bleu ${DATA_DIR}/dev.${SRC}-${TRG}.tok.${TRG} $CKPT_DIR/dev.${SRC}-${TRG}.tok.out.${TRG}

# Run on the test set
python $PTT_DIR/generate.py \
  --no-progress-bar \
  --path $CKPT_DIR/averaged_checkpoint_best.pt \
  --source-vocab-file $CKPT_DIR/dictionary-${SRC}.txt \
  --target-vocab-file $CKPT_DIR/dictionary-${TRG}.txt \
  --source-text-file ${DATA_DIR}/test.${SRC}-${TRG}.tok.${SRC} \
  --target-text-file ${DATA_DIR}/test.${SRC}-${TRG}.tok.${TRG} \
  --beam 5 \
  --length-penalty 1.0 \
  --max-tokens 2000 \
  --quiet \
  --translation-output-file $CKPT_DIR/test.${SRC}-${TRG}.tok.out.${TRG}

# Compute test BLEU score
echo "Detokenized BLEU score on the test set"
bleu ${DATA_DIR}/test.${SRC}-${TRG}.tok.${TRG} $CKPT_DIR/test.${SRC}-${TRG}.tok.out.${TRG}

###########
# Attacks #
###########

# Attack dev set
bash scripts/perform-attacks.sh \
  $SRC \
  $TRG \
  $DATA_DIR/dev.${SRC}-${TRG}.tok \
  $CKPT_DIR \
  5 \
  1 \
  "gold" \
  "--beam 5 --length-penalty 1.0"

# Post-process unk-only
bash scripts/make-unk-attacks-human-readable.sh $CKPT_DIR/attacks_results.zip $CKPT_DIR/dictionary-$SRC.txt $SRC > /dev/null 2>&1
# Print BLEU
echo "BLEU scores"
bash scripts/attacks-bleu-scores.sh $CKPT_DIR/attacks_results.zip
# Compute and print METEOR
echo "METEOR scores"
bash  scripts/attacks-meteor-scores.sh $CKPT_DIR/attacks_results.zip $SRC $TRG




