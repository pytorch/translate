#!/bin/bash
# Builds PyTorch Translate and runs basic tests.

pip uninstall -y pytorch-translate
python3 setup.py build develop
python3 setup.py test

# TODO(weiho): Re-enable testing these end-to-end scripts after refactoring
#     out the wget to be part of the Dockerfile. Possibly wait for v2 of our
#     OSS CI.
# . pytorch_translate/examples/train_iwslt14.sh
# . pytorch_translate/examples/generate_iwslt14.sh
# . pytorch_translate/examples/export_iwslt14.sh
# echo "hallo welt ." | . pytorch_translate/examples/translate_iwslt14.sh
