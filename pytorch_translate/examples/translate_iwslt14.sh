#!/bin/bash
#
# Sample script to run the decoder to load an exported model and use it for
# inference. Assumes that `install.sh` has been run, so that `cmake` and
# `make` have already been run in the `pytorch_translate/cpp` directory,
# producing the `translation_decoder` binary.
#
# Sample usage:
# echo "hallo welt ." | bash pytorch_translate/examples/translate_iwslt14.sh

cat | pytorch_translate/cpp/translation_decoder \
  --encoder_model "encoder.pb" \
  --decoder_step_model "decoder.pb" \
  --source_vocab_path "model/dictionary-de.txt" \
  --target_vocab_path "model/dictionary-en.txt" \
  `# Tuneable parameters` \
  --beam_size 6 \
  --max_out_seq_len_mult 1.1 \
  --max_out_seq_len_bias 5 \
  `# Must match your training settings` \
  --reverse_source True \
  --append_eos_to_source False \
  `# Unset for more logging/debug messages` \
  --caffe2_log_level 3
