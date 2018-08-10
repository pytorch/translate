#!/bin/bash

# Retrieve global variables
source scripts/globals.sh

#############
# Arguments #
#############

SRC_LANG=$1           # Source language
TRG_LANG=$2           # Target language
DATA_PREFIX=$3        # Prefix for the data files
CKPT_DIR=$4           # Checkpoint dir (containing averaged_checkpoint_best.pt)
MAX_ITERS=$5          # Maximum attack iterations
MAX_SWAPS=$6          # Maximum swaps allowed during each attack
DEVIATE_FROM=$7       # Which target sentence to compute the adversarial objective on (either "gold" or "generated")
TRANSLATE_OPTIONS=$8  # Options passed to the generate.py script (typically beam size, length penalty, etc..)

#####################
# Global parameters #
#####################

# Unique id for temp files
TMP_ID=`uuidgen`

# Experiment name (make it unique)
FULL_EXP_NAME="iwslt2016.${SRC_LANG}-${TRG_LANG}.attack"
TMP_DIR=/tmp/$FULL_EXP_NAME.$TMP_ID
echo "Temp dir: $TMP_DIR"
mkdir -p $TMP_DIR

# Model files
MODEL_FILE=$CKPT_DIR/averaged_checkpoint_best.pt
SRC_DICT=$CKPT_DIR/dictionary-${SRC_LANG}.txt
TRG_DICT=$CKPT_DIR/dictionary-${TRG_LANG}.txt

# Attacks: these are all the criterions/objectives we'll test
declare -A ATTACKS

ATTACKS["random"]="--adversary random_swap --adv-criterion cross_entropy"
ATTACKS["all_wrong"]="--adversary brute_force --adv-criterion all_bad_words --modify-gradient sign"
ATTACKS["force_long"]="--adversary brute_force --adv-criterion force_words --force-not --words-list </s> --modify-gradient sign"

# Constraints: each attack will be tested under those 3 constraints
declare -A CONSTRAINTS
CONSTRAINTS["unconstrained"]=""
CONSTRAINTS["nearest-neighbors"]="--nearest-neighbors 10"
CONSTRAINTS["unk-only"]="--allowed-tokens <unk>"

# Log file
LOG_FILE="$TMP_DIR/attack_log.txt"

######################
# Copy files locally #
######################

# Copy data
cp ${DATA_PREFIX}.$SRC_LANG $TMP_DIR/data.$SRC_LANG
cp ${DATA_PREFIX}.$TRG_LANG $TMP_DIR/data.$TRG_LANG

# For debugging
ls -lh $TMP_DIR/*

####################
# Helper functions #
####################

# Produces a hash from an experiment name, used to get fixed random seed
# Courtesy of https://stackoverflow.com/a/7265130
function get_seed (){
  # Hash
  n=$(md5sum <<< "$1")
  # Convert to decimals
  n=$((0x${n%% *}))
  # Take the absolute value
  echo ${n#-}
}

# Generate adversarial inputs
function attack (){
  ATTACK_ARGS=$1
  SRC_FILE=$2
  TRG_FILE=$3
  OUT_FILE=$4
  # Get a good random seed
  ATTACK_STRING="$FULL_EXP_NAME $ATTACK_ARGS $SRC_FILE $TRG_FILE $OUT_FILE"
  SEED=$(get_seed "$ATTACK_STRING")
  # Run the attack
  python $PTT_DIR/research/adversarial/whitebox.py \
    --no-progress-bar \
    --path $MODEL_FILE \
    --source-vocab-file $SRC_DICT \
    --target-vocab-file $TRG_DICT \
    --source-text-file $SRC_FILE \
    --target-text-file $TRG_FILE \
    $ATTACK_ARGS \
    --quiet \
    --seed $SEED \
    --max-sentences 32 \
    --max-tokens 4000 \
    --adversarial-output-file $OUT_FILE
}

# Translate using generate.py
function translate (){
  SRC_FILE=$1
  OUT_FILE=$2
  TRANSLATE_ARGS=$3
  python $PTT_DIR/generate.py \
    --no-progress-bar \
    --path $MODEL_FILE \
    --source-vocab-file $SRC_DICT \
    --target-vocab-file $TRG_DICT \
    --source-text-file $SRC_FILE \
    --target-text-file $TMP_DIR/data.${TRG_LANG} \
    $TRANSLATE_OPTIONS \
    --quiet \
    --max-sentences 64 \
    --max-tokens 4000 \
    --translation-output-file "${OUT_FILE}"
}

# Compute BLEU with uniform tokenization
function bleu (){
  REF_FILE=$1
  HYP_FILE=$2
  # Detokenize/de-bpe first just in case
  cat $REF_FILE | sed -r 's/(@@ )|(@@ ?$)//g' | \
    perl $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $TRG_LANG \
    > /tmp/gold.${TMP_ID}.txt
  cat $HYP_FILE | sed -r 's/(@@ )|(@@ ?$)//g' | \
    perl $MOSES_SCRIPTS/tokenizer/detokenizer.perl -l $TRG_LANG \
    > /tmp/hyp.${TMP_ID}.txt
  # Bleu
  perl $MOSES_SCRIPTS/generic/multi-bleu-detok.perl \
    /tmp/gold.${TMP_ID}.txt < /tmp/hyp.${TMP_ID}.txt
}

# Count the occurences of a certain word
function word_count (){
  FILE=$1
  for word in `echo $WORDS_LIST`; do
    echo -n "# $word: "
    sed 's/ /\n/g' $FILE | grep -ic "\b$word\b"
  done
}

##############
# Experiment #
##############

echo -n "" > $LOG_FILE

# Translate actual source file (for comparison)
src_file="$TMP_DIR/data.${SRC_LANG}"
out_file="$TMP_DIR/data.base.${TRG_LANG}"
translate $src_file $out_file $TRANSLATE_OPTIONS

# Get BLEU
THIS_BLEU=`bleu $TMP_DIR/data.${TRG_LANG} ${out_file} | grep "BLEU = "`
echo "Baseline: " >> $LOG_FILE
echo $THIS_BLEU >> $LOG_FILE

# Iterate over number of swaps
for ((swaps=1; swaps<=MAX_SWAPS; swaps++))
do
  # Iterate over attacks
  for atk in "${!ATTACKS[@]}"
  do
    # And constraints
    for constraint in "${!CONSTRAINTS[@]}"
    do
      attack_name="${atk}.${constraint}"
      # Retrieve command-line arguments for the attack/constraint
      attack_args=${ATTACKS[$atk]}
      constraint_args=${CONSTRAINTS[$constraint]}
      whitebox_args="$attack_args $constraint_args --max-swaps $swaps"

      # Start from the source file and the generated output
      attack_src_file="$TMP_DIR/data.${SRC_LANG}"
      attack_trg_file="$TMP_DIR/data.base.${TRG_LANG}"
      # Start iterating
      for ((iter=1; iter<=MAX_ITERS; iter++))
      do
        # Determine target file for the attack
        if [ $DEVIATE_FROM == "gold" ]
        then
          # Actually let's use the reference as a target from the attack
          attack_trg_file="$TMP_DIR/data.${TRG_LANG}"
        fi
        # Output file
        prefix=${attack_name}.${swaps}.${iter}
        attack_out_file=$TMP_DIR/data.${prefix}.${SRC_LANG}
        # Attack
        attack "$whitebox_args" $attack_src_file $attack_trg_file $attack_out_file
        # Translate
        out_file="$TMP_DIR/data.${prefix}.${TRG_LANG}"
        translate $attack_out_file $out_file $TRANSLATE_OPTIONS
        # Eval BLEU
        THIS_BLEU=`bleu $TMP_DIR/data.${TRG_LANG} $out_file | grep "BLEU = "`
        echo "-----------------" >> $LOG_FILE
        echo "${prefix}: " >> $LOG_FILE
        echo $THIS_BLEU >> $LOG_FILE

        # Update source/target file for the attack
        attack_src_file=$attack_out_file
        attack_trg_file=$out_file
      done
    done
  done
done

# Show the log
cat $LOG_FILE

# Compress everything
zip -j $CKPT_DIR/attacks_results.zip $TMP_DIR/*
rm -r $TMP_DIR
