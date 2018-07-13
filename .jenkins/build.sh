#!/bin/bash
# Builds PyTorch Translate and runs basic tests.

pip uninstall -y pytorch-translate
python3 setup.py build develop
pushd pytorch_translate/cpp || exit

mkdir build && pushd build || exit
cmake \
  -DCMAKE_PREFIX_PATH="${CONDA_PATH}/usr/local" \
  -DCMAKE_INSTALL_PREFIX="${CONDA_PATH}" .. \
  2>&1 | tee CMAKE_OUT
make 2>&1 | tee MAKE_OUT
# Return to the translate directory.
popd || exit
popd || exit
python3 setup.py test

# TODO(weiho): Re-enable testing these end-to-end scripts after refactoring
#     out the wget to be part of the Dockerfile. Possibly wait for v2 of our
#     OSS CI.
# . pytorch_translate/examples/train_iwslt14.sh
# . pytorch_translate/examples/generate_iwslt14.sh
# . pytorch_translate/examples/export_iwslt14.sh
# echo "hallo welt ." | . pytorch_translate/examples/translate_iwslt14.sh
