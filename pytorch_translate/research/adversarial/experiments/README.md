# Adversarial MT experiments

1. [Requirements](#requirements)
2. [Data download](#data-download)
3. [Experiment walkthrough](#experiment-walkthrough)
4. [Full experiments](#full-experiments)

## Requirements

For the data preprocessing, you will need Moses (tokenization/detokenization/BLEU) and subword-nmt (BPE):

```bash
git clone https://github.com/moses-smt/mosesdecoder.git
pip install subword-nmt
```

Other than that you should have a working version of `pytorch_translate` with pytorch 0.4.1. All scripts assume that you are running them from this directory.

## Data download

```bash
bash prepare-iwslt2016.sh
```

will download the data for IWSLT2016 in three language pairs (`fr-en`, `de-en`, `cs-en`), tokenize it and apply BPE.

## Experiment walkthrough

Here is an example usage for the LSTM model on  `cs-en` (the smallest language pair).

First, train the model

```bash
# Create the checkpoint folder
mkdir -p checkpoints/
mkdir -p checkpoints/iwslt2016-cs-en-lstm-base
# Create the folder for the binarized dataset
mkdir -p iwslt2016/cs-en

# Train
python ../../../train.py \
  --log-verbose  \
  --args-verbosity 2 \
  --no-progress-bar  \
  --arch rnn \
  --cell-type lstm \
  --sequence-lstm  \
  --max-tokens 5000 \
  --max-sentences 64 \
  --max-epoch 999999 \
  --stop-time-hr 140 \
  --optimizer adam \
  --lr 0.001 \
  --lr-shrink 0.5 \
  --clip-norm 5.0 \
  --encoder-dropout-in 0.3 \
  --encoder-dropout-out 0.3 \
  --decoder-dropout-in 0.3 \
  --decoder-dropout-out 0.3 \
  --criterion "label_smoothed_cross_entropy" \
  --label-smoothing 0.1 \
  --encoder-layers 2 \
  --encoder-embed-dim 300 \
  --encoder-hidden-dim 500 \
  --decoder-layers 2 \
  --decoder-embed-dim 300 \
  --decoder-hidden-dim 500 \
  --decoder-out-embed-dim 300 \
  --attention-type dot \
  --sentence-avg  \
  --momentum 0 \
  --generate-bleu-eval-per-epoch  \
  --beam 5 \
  --no-beamable-mm  \
  --max-len-a 1.2 \
  --max-len-b 10 \
  --source-max-vocab-size 40000 \
  --target-max-vocab-size 40000 \
  --log-interval 500 \
  --seed 42 \
  --decoder-tie-embeddings  \
  --max-checkpoints-kept 1 \
  --source-lang cs \
  --target-lang en \
  --save-dir checkpoints/iwslt2016-cs-en-lstm-base \
  --train-source-text-file iwslt2016/cs-en/train.cs-en.tok.cs \
  --train-target-text-file iwslt2016/cs-en/train.cs-en.tok.en \
  --eval-source-text-file iwslt2016/cs-en/dev.cs-en.tok.cs \
  --eval-target-text-file iwslt2016/cs-en/dev.cs-en.tok.en \
  --train-source-binary-path iwslt2016/cs-en/train.cs-en.cs \
  --train-target-binary-path iwslt2016/cs-en/train.cs-en.en \
  --eval-source-binary-path iwslt2016/cs-en/dev.cs-en.cs \
  --eval-target-binary-path iwslt2016/cs-en/dev.cs-en.en
```

Now evaluate the model on the validation and test sets

```bash
# This is a helper function to compute BLEU score with uniform tokenization
function bleu (){
  REF_FILE=$1
  HYP_FILE=$2
  # Detokenize/de-bpe first just in case
  cat $REF_FILE | sed -r 's/(@@ )|(@@ ?$)//g' | \
    perl mosesdecoder/scripts/tokenizer/detokenizer.perl -l en \
    > /tmp/gold.txt
  cat $HYP_FILE | sed -r 's/(@@ )|(@@ ?$)//g' | \
    perl mosesdecoder/scripts/tokenizer/detokenizer.perl -l en \
    > /tmp/hyp.txt
  # Bleu
  perl mosesdecoder/scripts/generic/multi-bleu-detok.perl \
    /tmp/gold.txt < /tmp/hyp.txt
}

# Run on the dev set
python ../../../generate.py \
  --no-progress-bar \
  --path checkpoints/iwslt2016-cs-en-lstm-base/averaged_checkpoint_best.pt \
  --source-vocab-file checkpoints/iwslt2016-cs-en-lstm-base/dictionary-cs.txt \
  --target-vocab-file checkpoints/iwslt2016-cs-en-lstm-base/dictionary-en.txt \
  --source-text-file iwslt2016/cs-en/dev.cs-en.tok.cs \
  --target-text-file iwslt2016/cs-en/dev.cs-en.tok.en \
  --beam 5 \
  --max-tokens 2000 \
  --quiet \
  --translation-output-file checkpoints/iwslt2016-cs-en-lstm-base/dev.cs-en.tok.out.en

# Compute dev BLEU score
echo "Detokenized BLEU score on the dev set"
bleu iwslt2016/cs-en/dev.cs-en.tok.en checkpoints/iwslt2016-cs-en-lstm-base/dev.cs-en.tok.out.en

# Run on the test set
python ../../../generate.py \
  --no-progress-bar \
  --path checkpoints/iwslt2016-cs-en-lstm-base/averaged_checkpoint_best.pt \
  --source-vocab-file checkpoints/iwslt2016-cs-en-lstm-base/dictionary-cs.txt \
  --target-vocab-file checkpoints/iwslt2016-cs-en-lstm-base/dictionary-en.txt \
  --source-text-file iwslt2016/cs-en/test.cs-en.tok.cs \
  --target-text-file iwslt2016/cs-en/test.cs-en.tok.en \
  --beam 5 \
  --max-tokens 2000 \
  --quiet \
  --translation-output-file checkpoints/iwslt2016-cs-en-lstm-base/test.cs-en.tok.out.en

# Compute test BLEU score
echo "Detokenized BLEU score on the test set"
bleu iwslt2016/cs-en/test.cs-en.tok.en checkpoints/iwslt2016-cs-en-lstm-base/test.cs-en.tok.out.en
```

Finally try some attacks on the model

```bash
# Attack unk-only on the dev set
python ../whitebox.py \
  --no-progress-bar \
  --path checkpoints/iwslt2016-cs-en-lstm-base/averaged_checkpoint_best.pt \
  --source-vocab-file checkpoints/iwslt2016-cs-en-lstm-base/dictionary-cs.txt \
  --target-vocab-file checkpoints/iwslt2016-cs-en-lstm-base/dictionary-en.txt \
  --source-text-file iwslt2016/cs-en/dev.cs-en.tok.cs \
  --target-text-file iwslt2016/cs-en/dev.cs-en.tok.en \
  --adv-criterion all_bad_words \
  --adversary brute_force \
  --allowed-tokens "<unk>" \
  --quiet \
  --seed 42 \
  --max-sentences 32 \
  --max-tokens 4000 \
  --adversarial-output-file checkpoints/iwslt2016-cs-en-lstm-base/dev.cs-en.tok.adv-unk.cs

# Try to translate the adversarial inputs
python ../../../generate.py \
  --no-progress-bar \
  --path checkpoints/iwslt2016-cs-en-lstm-base/averaged_checkpoint_best.pt \
  --source-vocab-file checkpoints/iwslt2016-cs-en-lstm-base/dictionary-cs.txt \
  --target-vocab-file checkpoints/iwslt2016-cs-en-lstm-base/dictionary-en.txt \
  --source-text-file checkpoints/iwslt2016-cs-en-lstm-base/dev.cs-en.tok.adv-unk.cs \
  --target-text-file iwslt2016/cs-en/dev.cs-en.tok.en \
  --beam 5 \
  --max-tokens 2000 \
  --quiet \
  --translation-output-file checkpoints/iwslt2016-cs-en-lstm-base/dev.cs-en.tok.adv-unk.out.en

# Get the BLEU score
echo "Detokenized BLEU score on the peturbated dev set"
bleu iwslt2016/cs-en/dev.cs-en.tok.en checkpoints/iwslt2016-cs-en-lstm-base/dev.cs-en.tok.adv-unk.out.en
```

You can do some more analysis using some of the provided scripts

```bash
# First de-escape special chars (like apostrophes) because we're going to 
# do character level manipulations
cat checkpoints/iwslt2016-cs-en-lstm-base/dev.cs-en.tok.adv-unk.cs \
  | perl $MOSES_SCRIPTS/tokenizer/deescape-special-chars.perl \
  > checkpoints/iwslt2016-cs-en-lstm-base/dev.cs-en.tok.adv-unk.deescaped.cs
# Post-process unk-only: this will replace the `<unk>` introduced by the 
# adversarial attack with actual typos (human readable). Those will still
# be unknown words for the model
python scripts/generate_unk_adversaries.py \
  iwslt2016/cs-en/dev.cs-en.tok.cs \
  checkpoints/iwslt2016-cs-en-lstm-base/dev.cs-en.tok.adv-unk.deescaped.cs \
  --dictionary-file checkpoints/iwslt2016-cs-en-lstm-base/dictionary-cs.txt \
  > checkpoints/iwslt2016-cs-en-lstm-base/dev.cs-en.tok.adv-unk.human-readable.cs
# Checkout some examples
head checkpoints/iwslt2016-cs-en-lstm-base/dev.cs-en.tok.adv-unk.human-readable.cs
# Compute and print METEOR
echo "Source METEOR scores"
java -Xmx2G -jar meteor-1.5/meteor-1.5.jar \
  checkpoints/iwslt2016-cs-en-lstm-base/dev.cs-en.tok.adv-unk.human-readable.cs \
  iwslt2016/cs-en/dev.cs-en.tok.cs \
  -l cz -norm -q \
  2> /dev/null
```

## Full experiments

Each of the `iwslt2016-[src_lang]-en-[model_type]-[training]-full.sh` file contains all experiments (training, testing, attacks) for the model where:

- `src_lang` is the source language (one of `cs`, `de`, `fr`)
- `model_type` is one of `lstm` or `transformer`
- `training` is one of
  - `base`: normal training
  - `rand-unk`: "adversarial" training where each sample is perturbated by adding an `<unk>` at random
  - `adv-unk`: proper adversarial training, each samples is perturbated by adding an `<unk>` chosen with a whitebox adversarial attack.
