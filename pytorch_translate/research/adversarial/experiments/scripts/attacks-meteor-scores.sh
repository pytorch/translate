#!/bin/bash

# Retrieve global variables
source scripts/globals.sh

# Zip file produced by "scripts/perform-attacks"
ZIP_FILE=$1
dirname $ZIP_FILE
# Source and target languages
SRC=$2
TRG=$3
# this is the suffix indicating [max_swaps].[max_iters]
SUFFIX=${4:-"1.1"}

UNIQUE_ID=`uuidgen`

# Helper function running the meteor
function meteor () {
  # cs -> cz for meteor ??? what about ISO 639-1 ?
  if [ $3 == "cs" ]
  then
    lang="cz"
  else
    lang=$3
  fi
  java -Xmx2G -jar $METEOR_DIR/meteor-1.5.jar $1 $2 -l $lang -norm -q 2> /dev/null
}

# Extract reference source & target
unzip -c $ZIP_FILE data.$SRC > /tmp/data.$UNIQUE_ID.$SRC
unzip -c $ZIP_FILE data.$TRG > /tmp/data.$UNIQUE_ID.$TRG
# Extract output from the model
unzip -c $ZIP_FILE data.base.$TRG > /tmp/data.base.$UNIQUE_ID.$TRG

# Meteor of "original source vs original source" for consistency
echo "1.0"

# Source METEOR for all constraints/attacks
for constraint in "unconstrained" "nearest-neighbors" "unk-only"
do
  for attack in "random" "all_wrong" "force_long"
  do
    prefix="$attack.$constraint"
    # Extract adversarial source
    if [ $constraint == "unk-only" ]
    then
      # For UnkOnly we use the human_readable post edited versions with OOV actual typos
      unzip -c $ZIP_FILE data.$prefix.$SUFFIX.$SRC.human_readable > /tmp/data.$prefix.$UNIQUE_ID.$SRC
    else
      unzip -c $ZIP_FILE data.$prefix.$SUFFIX.$SRC > /tmp/data.$prefix.$UNIQUE_ID.$SRC
    fi

    meteor /tmp/data.$prefix.$UNIQUE_ID.$SRC /tmp/data.$UNIQUE_ID.$SRC $SRC
  done
done

# Meteor of the model's output on the original source
meteor /tmp/data.base.$UNIQUE_ID.$TRG /tmp/data.$UNIQUE_ID.$TRG $TRG
# Target METEOR for all constraints/attacks
for constraint in "unconstrained" "nearest-neighbors" "unk-only"
do
  for attack in "random" "all_wrong" "force_long"
  do
    prefix="$attack.$constraint"
    # Extract output of the model for this attack
    unzip -c $ZIP_FILE data.$prefix.$SUFFIX.$TRG > /tmp/data.$prefix.$UNIQUE_ID.$TRG

    meteor /tmp/data.$prefix.$UNIQUE_ID.$TRG /tmp/data.$UNIQUE_ID.$TRG $TRG
  done
done
