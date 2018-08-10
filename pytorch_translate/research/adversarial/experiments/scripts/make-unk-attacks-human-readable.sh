#!/bin/bash

# Retrieve global variables
source scripts/globals.sh

# Zip file produced by "scripts/perform-attacks"
ZIP_FILE=$1
dirname $ZIP_FILE
# Source dictionary (to identify UNKs)
DICT_FILE=$2
# Source languages
SRC=$3

# Unique ID for temp files
UNIQUE_ID=`uuidgen`

# Temp dir
TMP_DIR=/tmp/unk-readable-$UNIQUE_ID
mkdir $TMP_DIR

# Extract all unk inputs
unzip $ZIP_FILE -d $TMP_DIR

# De-escape special chars to make sure that punctuation is
# handled correctly at the character level (eg &apos; -> ')
cat $TMP_DIR/data.$SRC \
  | perl $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl \
  > /tmp/$UNIQUE_ID.src.tok

for unk_src in $TMP_DIR/*unk-only.*.$SRC
do
  # De-escape special chars to make sure that punctuation is
  # handled correctly at the character level (eg &apos; -> '
  cat $unk_src | perl $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl \
    > /tmp/$UNIQUE_ID.adv.tok

  # Replace unks and re-escape special chars
  python scripts/unk-to-typo.py \
    /tmp/$UNIQUE_ID.src.tok \
    /tmp/$UNIQUE_ID.adv.tok \
    --dictionary-file $DICT_FILE \
    | perl $MOSES_SCRIPTS/tokenizer/escape-special-chars.perl \
    > $unk_src.human_readable
done

# Add the *.human_readable files to the archive
zip -j -u $ZIP_FILE $TMP_DIR/*.human_readable
