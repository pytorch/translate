#!/bin/bash

####################
# Global variables #
####################

# Path to pytorch_translate root (where `train.py` is)
PTT_DIR="../../.."
# Path to moses script folder
# Install with `git clone https://github.com/moses-smt/mosesdecoder.git`
MOSES_SCRIPTS=mosesdecoder/scripts
# Command to call for subword-nmt
# (see https://github.com/rsennrich/subword-nmt#installation) for installation
SUBWORD_NMT=subword-nmt
# Meteor command location
# wget http://www.cs.cmu.edu/\~alavie/METEOR/download/meteor-1.5.tar.gz && tar xvzf meteor-1.5.tar.gz
METEOR_DIR=meteor-1.5
# Checkpoints folder 
# (will contain separate checkpoint folders for each experiment)
CHECKPOINTS=checkpoints
# Root of the IWSLT2016 data downloaded with scripts/prepare-iwslt2016.sh
# Should contain one sub folder for each language pair
IWSLT2016_DIR=iwslt2016
