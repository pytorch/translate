#!/bin/bash

# Retrieve global variables
source scripts/globals.sh

# Zip file produced by "scripts/perform-attacks"
ZIP_FILE=$1
dirname $ZIP_FILE
# this is the suffix indicating [max_swaps].[max_iters]
SUFFIX=${2:-"1.1"}

# Extract log file
TMP_LOG=/tmp/log.`uuidgen`.txt
unzip -c $ZIP_FILE *attack_log.txt > $TMP_LOG

# Print baseline BLEU
grep "Baseline" $TMP_LOG -A1 | awk '{if ($1=="BLEU") print $3}'| sed 's/[^0-9\.]//g'

# Print BLEUs for all constraints/attacks
for constraint in "unconstrained" "nearest-neighbors" "unk-only"
do
  for attack in "random" "all_wrong" "force_long"
  do
    grep "$attack.$constraint.$SUFFIX:" $TMP_LOG -A1 | awk '{if ($1=="BLEU") print $3}' | sed 's/[^0-9\.]//g'
  done
done
