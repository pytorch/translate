#!/bin/bash

if [ -n "$(which conda)" ]; then
    pushd ~
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p ~/miniconda
    rm miniconda.sh
    popd

    . ~/miniconda/bin/activate

    export CONDA_PATH="$(dirname $(which conda))/../" # [anaconda root directory]
fi

# Install basic PyTorch dependencies
yes | conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing

# Add LAPACK support for the GPU
yes | conda install -c pytorch magma-cuda80 # or magma-cuda90 if CUDA 9

yes | pip uninstall torch
yes | pip uninstall torch

# Install NCCL2
wget https://s3.amazonaws.com/pytorch/nccl_2.1.15-1%2Bcuda8.0_x86_64.txz
tar --no-same-owner -xvf nccl_2.1.15-1+cuda8.0_x86_64.txz
export NCCL_ROOT_DIR="$(pwd)/nccl_2.1.15-1+cuda8.0_x86_64"
export LD_LIBRARY_PATH="${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
rm nccl_2.1.15-1+cuda8.0_x86_64.txz

git clone --recursive https://github.com/pytorch/pytorch
pushd pytorch
# PyTorch build from source
NCCL_ROOT_DIR="${NCCL_ROOT_DIR}" python3 setup.py install

# Caffe2 relies on past module
yes | pip install future

# Caffe2 build from source (with ATen)
mkdir -p build_caffe2 && pushd build_caffe2
cmake \
  -DPYTHON_INCLUDE_DIR=$(python -c 'from distutils import sysconfig; print(sysconfig.get_python_inc())') \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DUSE_ATEN=ON -DUSE_OPENCV=OFF \
  -DCMAKE_PREFIX_PATH="${CONDA_PATH}" -DCMAKE_INSTALL_PREFIX="${CONDA_PATH}" ..
make install -j8 2>&1 | tee MAKE_OUT
popd
popd

export LD_LIBRARY_PATH="${CONDA_PATH}/lib:${LD_LIBRARY_PATH}"

# Install ONNX
git clone --recursive https://github.com/onnx/onnx.git
pip install ./onnx

yes | pip uninstall pytorch-translate
python3 setup.py build develop

pushd pytorch_translate/cpp
# If you need to specify a compiler other than the default one cmake is picking
# up, you can use the -DCMAKE_C_COMPILER and -DCMAKE_CXX_COMPILER flags.
cmake -DCMAKE_PREFIX_PATH="${CONDA_PATH}/usr/local" -DCMAKE_INSTALL_PREFIX="${CONDA_PATH}" .
make
popd
