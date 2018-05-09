#!/bin/bash
#
# Sample script to run the decoder to load an exported model and use it for
# inference. Assumes that `install.sh` has been run, so that `cmake` and
# `make` have already been run in the `pytorch_translate/cpp` directory,
# producing the `translation_decoder` binary.
#
# Sample usage:
# echo "hallo welt" | bash pytorch_translate/examples/translate_iwslt14.sh

CONDA_PATH="$(dirname "$(which conda)")/../"
export CONDA_PATH
NCCL_ROOT_DIR="$(pwd)/nccl_2.1.15-1+cuda8.0_x86_64"
export NCCL_ROOT_DIR
LD_LIBRARY_PATH="${CONDA_PATH}/lib:${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH

cat | pytorch_translate/cpp/build/translation_decoder \
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
