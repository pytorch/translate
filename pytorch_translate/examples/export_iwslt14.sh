#!/bin/bash

. ~/miniconda/bin/activate
export NCCL_ROOT_DIR="$(pwd)/nccl_2.1.15-1+cuda8.0_x86_64"
export LD_LIBRARY_PATH="${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
wget https://download.pytorch.org/models/translate/iwslt14/model.tar.gz https://download.pytorch.org/models/translate/iwslt14/data.tar.gz
tar -xvzf model.tar.gz
tar -xvzf data.tar.gz
rm -f data.tar.gz model.tar.gz
python3 pytorch_translate/onnx_component_export.py \
    --checkpoint model/averaged_checkpoint_best_0.pt \
    --encoder_output_file encoder.pb \
    --decoder_output_file decoder.pb \
    --src_dict model/dictionary-de.txt \
    --dst_dict model/dictionary-en.txt \
    --beam_size 6 \
    --word_penalty 0.25 \
    --unk_penalty -0.5 \
    --batched_beam
