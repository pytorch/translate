#!/bin/bash

NCCL_ROOT_DIR="$(pwd)/nccl_2.1.15-1+cuda8.0_x86_64"
export NCCL_ROOT_DIR
LD_LIBRARY_PATH="${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
wget https://download.pytorch.org/models/translate/iwslt14/model.tar.gz https://download.pytorch.org/models/translate/iwslt14/data.tar.gz
tar -xvzf model.tar.gz
tar -xvzf data.tar.gz
rm -f data.tar.gz model.tar.gz
python3 pytorch_translate/onnx_component_export.py \
    --checkpoint model/averaged_checkpoint_best_0.pt \
    --encoder-output-file encoder.pb \
    --decoder-output-file decoder.pb \
    --source-vocab-file model/dictionary-de.txt \
    --target-vocab-file model/dictionary-en.txt \
    --beam-size 6 \
    --word-reward 0.25 \
    --unk-reward -0.5 \
    --batched-beam && \
  echo "Finished exporting encoder as ./encoder.pb and decoder as ./decoder.pb"
