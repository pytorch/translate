#!/bin/bash

pushd ~
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p ~/miniconda
rm miniconda.sh
popd

. ~/miniconda/bin/activate

export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic PyTorch dependencies
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing

# Add LAPACK support for the GPU
conda install -c pytorch magma-cuda80 # or magma-cuda90 if CUDA 9

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# PyTorch build from source
python setup.py install

# Caffe2 build from source (with ATen)
CMAKE_ARGS=-DUSE_ATEN=ON python setup_caffe2.py install

# Install ONNX
git clone https://github.com/onnx/onnx.git
pip install onnx

yes | pip uninstall fbtranslate
python3 setup.py build develop
