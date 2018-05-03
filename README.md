# Translate - a PyTorch Language Library

Translate is a library for machine translation written in PyTorch. It provides training for sequence-to-sequence models. Translate relies on [fairseq](https://github.com/pytorch/fairseq), a general sequence-to-sequence library, which means that models implemented in both Translate and Fairseq can be trained. Translate also provides the ability to export some models to Caffe2 graphs via [ONNX](https://onnx.ai/) and to load and run these models from C++ for production purposes. Currently, we export components (encoder, decoder) to Caffe2 separately and beam search is implemented in C++. In the near future, we will be able to export the beam search as well. We also plan to add export support to more models.

## Requirements and Installation

### Translate requires:
* A Linux operating system with a CUDA compatible card
* C++ compiler supporting ECMAScript syntax for `<regex>`, such as GCC 4.9 and above
* A [CUDA installation](https://docs.nvidia.com/cuda/). We recommend CUDA 8 or CUDA 9

### To install Translate from source:
These instructions were mainly tested on CentOS 7.4.1708 with a Tesla M40 card
and a CUDA 8 installation. We highly encourage you to [report an issue](https://github.com/pytorch/translate/issues)
if you are unable to install this project for your specific configuration.

- If you don't already have an existing [Anaconda](https://www.anaconda.com/download/)
environment with Python 3.6, you can install one via [Miniconda3](https://conda.io/miniconda.html):
  ```bash
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  chmod +x miniconda.sh
  ./miniconda.sh -b -p ~/miniconda
  rm miniconda.sh
  . ~/miniconda/bin/activate
  ```

- Clone the Translate repo:
  ```bash
  git clone https://github.com/pytorch/translate.git
  pushd translate
  ```

- Build [PyTorch](https://pytorch.org/) from source (currently needed for ONNX compatibility):
  ```bash
  # Uninstall previous versions of PyTorch. Doing this twice is intentional.
  # Error messages about torch not being installed are benign.
  pip uninstall -y torch
  pip uninstall -y torch

  # Install basic PyTorch dependencies.
  conda install -y cffi cmake mkl mkl-include numpy pyyaml setuptools typing
  # Add LAPACK support for the GPU.
  conda install -y -c pytorch magma-cuda80 # or magma-cuda90 if CUDA 9

  # Install NCCL2.
  wget https://s3.amazonaws.com/pytorch/nccl_2.1.15-1%2Bcuda8.0_x86_64.txz
  tar -xvf nccl_2.1.15-1+cuda8.0_x86_64.txz
  export NCCL_ROOT_DIR="$(pwd)/nccl_2.1.15-1+cuda8.0_x86_64"
  export LD_LIBRARY_PATH="${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
  rm nccl_2.1.15-1+cuda8.0_x86_64.txz

  # Build PyTorch from source.
  git clone --recursive https://github.com/pytorch/pytorch
  pushd pytorch
  git submodule update --init
  NCCL_ROOT_DIR="${NCCL_ROOT_DIR}" python3 setup.py install
  ```

- Build [Caffe2](http://caffe2.ai/) from source (under PyTorch):
  ```bash
  # Caffe2 relies on the past module.
  yes | pip install future

  export CONDA_PATH="$(dirname $(which conda))/.."

  # Compile Caffe2 from source with ATen.
  # If you need to specify a compiler other than the default one cmake is picking
  # up, you can use the -DCMAKE_C_COMPILER and -DCMAKE_CXX_COMPILER flags.
  mkdir build_caffe2 && pushd build_caffe2
  cmake \
    -DPYTHON_INCLUDE_DIR=$(python -c 'from distutils import sysconfig; print(sysconfig.get_python_inc())') \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DUSE_ATEN=ON \
    -DUSE_OPENCV=OFF \
    -DCMAKE_PREFIX_PATH="${CONDA_PATH}" \
    -DCMAKE_INSTALL_PREFIX="${CONDA_PATH}" .. \
    2>&1 | tee CMAKE_OUT
  make install -j8 2>&1 | tee MAKE_OUT

  export ATEN_LIB="$(pwd)/caffe2/contrib/aten/aten/lib"
  export LD_LIBRARY_PATH="${ATEN_LIB}:${CONDA_PATH}/lib:${LD_LIBRARY_PATH}"

  # Return to the translate directory.
  popd
  popd
  ```

- Install [ONNX](https://onnx.ai/):
  ```bash
  git clone --recursive https://github.com/onnx/onnx.git
  yes | pip install ./onnx
  ```

- Build Translate:
  ```bash
  pip uninstall -y pytorch-translate
  python3 setup.py build develop
  pushd pytorch_translate/cpp

  # If you need to specify a compiler other than the default one cmake is picking
  # up, you can use the -DCMAKE_C_COMPILER and -DCMAKE_CXX_COMPILER flags.
  mkdir build && pushd build
  cmake \
    -DCMAKE_PREFIX_PATH="${CONDA_PATH}/usr/local" \
    -DCMAKE_INSTALL_PREFIX="${CONDA_PATH}" .. \
    2>&1 | tee CMAKE_OUT
  make 2>&2 | tee MAKE_OUT

  # Return to the translate directory.
  popd
  popd
  ```
  
Now you should be able to run the example scripts below!

### To use our Amazon Machine Image:
You can launch an AWS instance using the `pytorch_translate_initial_release` image (AMI ID: ami-04ff53cdd573658dc). Once you have ssh'ed to the AWS instance, the example commands below should work after running `cd translate`.

## Training

Note: the example commands given assume that you are the root of the cloned GitHub repository or that you're using an AWS instance and that you have run `cd translate`.

We provide an [example script](https://github.com/pytorch/translate/blob/master/pytorch_translate/examples/train_iwslt14.sh) to train a model for the IWSLT 2014 German-English task. We used this command to obtain [a pretrained model](https://download.pytorch.org/models/translate/iwslt14/model.tar.gz):

```
bash pytorch_translate/examples/train_iwslt14.sh
```

The pretrained model actually contains two checkpoints that correspond to training twice with random initialization of the parameters. This is useful to obtain ensembles. This dataset is relatively small (~160K sentence pairs), so training will complete in a few hours on a single GPU.

## Pretrained Model

A pretrained model for IWSLT 2014 can be evaluated by running the [example script](https://github.com/pytorch/translate/blob/master/pytorch_translate/examples/generate_iwslt14.sh):

```
bash pytorch_translate/examples/generate_iwslt14.sh
```

Note the improvement in performance when using an ensemble of size 2 instead of a single model.

## Exporting a Model with ONNX

We provide an [example script](https://github.com/pytorch/translate/blob/master/pytorch_translate/examples/export_iwslt14.sh) to export a PyTorch model to a Caffe2 graph via ONNX:

```
bash pytorch_translate/examples/export_iwslt14.sh
```

This will output two files, `encoder.pb` and `decoder.pb`, that correspond to the computation of the encoder and one step of the decoder. The example exports a single checkpoint (`--checkpoint model/averaged_checkpoint_best_0.pt` but is also possible to export an ensemble (`--checkpoint model/averaged_checkpoint_best_0.pt --checkpoint model/averaged_checkpoint_best_1.pt`). Note that during export, you can also control a few hyperparameters such as beam search size, word and UNK rewards.

## Using the Model

To use the sample exported Caffe2 model to translate sentences, run:

```
echo "hallo welt ." | bash pytorch_translate/examples/translate_iwslt14.sh
```

Note that the model takes in [BPE](https://github.com/rsennrich/subword-nmt)
inputs, so some input words need to be split into multiple tokens.
For instance, "hineinstopfen" is represented as "hinein@@ stop@@ fen".

## Join the Translate Community

We welcome contributions! See the `CONTRIBUTING.md` file for how to help out.

## License
Translate is BSD-licensed, as found in the `LICENSE` file.
