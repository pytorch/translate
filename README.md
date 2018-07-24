# Translate - a PyTorch Language Library

trigger build

Translate is a library for machine translation written in PyTorch. It provides training for sequence-to-sequence models. Translate relies on [fairseq](https://github.com/pytorch/fairseq), a general sequence-to-sequence library, which means that models implemented in both Translate and Fairseq can be trained. Translate also provides the ability to export some models to Caffe2 graphs via [ONNX](https://onnx.ai/) and to load and run these models from C++ for production purposes. Currently, we export components (encoder, decoder) to Caffe2 separately and beam search is implemented in C++. In the near future, we will be able to export the beam search as well. We also plan to add export support to more models.

## Requirements and Installation

### Translate Requires:
* A Linux operating system with a CUDA compatible card
* GNU C++ compiler version 4.9.2 and above
* A [CUDA installation](https://docs.nvidia.com/cuda/). We recommend CUDA 8.0 or CUDA 9.0

### Use Our Docker Image:
Install [Docker](https://docs.docker.com/install/) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker), then run

```
sudo docker pull pytorch/translate
sudo nvidia-docker run -i -t --rm pytorch/translate /bin/bash
. ~/miniconda/bin/activate
cd ~/translate
```

You should now be able to run the sample commands in the
[Usage Examples](#usage-examples) section below. You can also see the available
image versions under https://hub.docker.com/r/pytorch/translate/tags/.

### Install Translate from Source:
These instructions were mainly tested on CentOS 7.4.1708 with a Tesla M40 card
and a CUDA 8 installation. We highly encourage you to [report an issue](https://github.com/pytorch/translate/issues)
if you are unable to install this project for your specific configuration.

- If you don't already have an existing [Anaconda](https://www.anaconda.com/download/)
environment with Python 3.6, you can install one via [Miniconda3](https://conda.io/miniconda.html):

  ```
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  chmod +x miniconda.sh
  ./miniconda.sh -b -p ~/miniconda
  rm miniconda.sh
  . ~/miniconda/bin/activate
  ```

- Clone the Translate repo:

  ```
  git clone https://github.com/pytorch/translate.git
  pushd translate
  ```

- Install the combined [PyTorch](https://pytorch.org/) and [Caffe2](http://caffe2.ai/) Conda [package](https://anaconda.org/caffe2):

  ```
  # Set to 8 or 9 depending on your CUDA version.
  TMP_CUDA_VERSION="9"

  # Uninstall previous versions of PyTorch. Doing this twice is intentional.
  # Error messages about torch not being installed are benign.
  pip uninstall -y torch
  pip uninstall -y torch

  # This may not be necessary if you already have the latest cuDNN library.
  conda install -y cudnn

  # Add LAPACK support for the GPU.
  conda install -y -c pytorch "magma-cuda${TMP_CUDA_VERSION}0"

  # Install the combined PyTorch+Caffe2 conda package.
  conda install -y -c caffe2 "pytorch-caffe2-cuda${TMP_CUDA_VERSION}.0-cudnn7"
  # Force re-install of numpy 1.14 since the current version of the
  # PyTorch+Caffe2 package downgrades it.
  conda install -y numpy==1.14 --no-deps

  # Install NCCL2.
  wget "https://s3.amazonaws.com/pytorch/nccl_2.1.15-1%2Bcuda${TMP_CUDA_VERSION}.0_x86_64.txz"
  TMP_NCCL_VERSION="nccl_2.1.15-1+cuda${TMP_CUDA_VERSION}.0_x86_64"
  tar -xvf "${TMP_NCCL_VERSION}.txz"
  rm "${TMP_NCCL_VERSION}.txz"

  # Set some environmental variables needed to link libraries correctly.
  export CONDA_PATH="$(dirname $(which conda))/.."
  export NCCL_ROOT_DIR="$(pwd)/${TMP_NCCL_VERSION}"
  export LD_LIBRARY_PATH="${CONDA_PATH}/lib:${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
  ```

- Install [ONNX](https://onnx.ai/):

  ```
  git clone --recursive https://github.com/onnx/onnx.git
  yes | pip install ./onnx 2>&1 | tee ONNX_OUT
  ```

- Build Translate:

  ```
  pip uninstall -y pytorch-translate
  python3 setup.py build develop
  pushd pytorch_translate/cpp

  mkdir build && pushd build
  cmake \
    -DCMAKE_PREFIX_PATH="${CONDA_PATH}/usr/local" \
    -DCMAKE_INSTALL_PREFIX="${CONDA_PATH}" .. \
    2>&1 | tee CMAKE_OUT
  make 2>&1 | tee MAKE_OUT
  # Return to the translate directory.
  popd
  popd
  ```

Now you should be able to run the example scripts below!

### Use Our Amazon Machine Image:
You can launch an AWS instance using the `pytorch_translate_initial_release` image (AMI ID: ami-04ff53cdd573658dc). Once you have ssh'ed to the AWS instance, the example commands below should work after running `cd translate`.

## Usage Examples

Note: the example commands given assume that you are the root of the cloned
GitHub repository or that you're in the `translate` directory of the Docker or
Amazon image. You may also need to make sure you have the Anaconda environment
activated.

### Training

We provide an [example script](https://github.com/pytorch/translate/blob/master/pytorch_translate/examples/train_iwslt14.sh) to train a model for the IWSLT 2014 German-English task. We used this command to obtain [a pretrained model](https://download.pytorch.org/models/translate/iwslt14/model.tar.gz):

```
bash pytorch_translate/examples/train_iwslt14.sh
```

The pretrained model actually contains two checkpoints that correspond to training twice with random initialization of the parameters. This is useful to obtain ensembles. This dataset is relatively small (~160K sentence pairs), so training will complete in a few hours on a single GPU.

####  Training with tensorboard visualization

We provide support for visualizing training stats with tensorboard. As a dependency, you will need [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger) installed.

```
pip install tensorboard_logger
```

Please also make sure that [tensorboard](https://github.com/tensorflow/tensorboard) is installed. It also comes with `tensorflow` installation.

You can use the above [example script](https://github.com/pytorch/translate/blob/master/pytorch_translate/examples/train_iwslt14.sh) to train with tensorboard, but need to change line 10 from :

```
CUDA_VISIBLE_DEVICES=0 python3 pytorch_translate/train.py
```
to

```
CUDA_VISIBLE_DEVICES=0 python3 pytorch_translate/train_with_tensorboard.py
```
The event log directory for tensorboard can be specified by option `--tensorboard_dir` with a default value: `run-1234`. This directory is appended to your `--save_dir` argument.

For example in the above script, you can visualize with:

```
tensorboard --logdir checkpoints/runs/run-1234
```

Multiple runs can be compared by specifying different `--tensorboard_dir`. i.e. `run-1234` and `run-2345`. Then

```
tensorboard --logdir checkpoints/runs
```

can visualize stats from both runs.

### Pretrained Model

A pretrained model for IWSLT 2014 can be evaluated by running the [example script](https://github.com/pytorch/translate/blob/master/pytorch_translate/examples/generate_iwslt14.sh):

```
bash pytorch_translate/examples/generate_iwslt14.sh
```

Note the improvement in performance when using an ensemble of size 2 instead of a single model.

### Exporting a Model with ONNX

We provide an [example script](https://github.com/pytorch/translate/blob/master/pytorch_translate/examples/export_iwslt14.sh) to export a PyTorch model to a Caffe2 graph via ONNX:

```
bash pytorch_translate/examples/export_iwslt14.sh
```

This will output two files, `encoder.pb` and `decoder.pb`, that correspond to the computation of the encoder and one step of the decoder. The example exports a single checkpoint (`--checkpoint model/averaged_checkpoint_best_0.pt` but is also possible to export an ensemble (`--checkpoint model/averaged_checkpoint_best_0.pt --checkpoint model/averaged_checkpoint_best_1.pt`). Note that during export, you can also control a few hyperparameters such as beam search size, word and UNK rewards.

### Using the Model

To use the sample exported Caffe2 model to translate sentences, run:

```
echo "hallo welt" | bash pytorch_translate/examples/translate_iwslt14.sh
```

Note that the model takes in [BPE](https://github.com/rsennrich/subword-nmt)
inputs, so some input words need to be split into multiple tokens.
For instance, "hineinstopfen" is represented as "hinein@@ stop@@ fen".

## Join the Translate Community

We welcome contributions! See the `CONTRIBUTING.md` file for how to help out.

## License
Translate is BSD-licensed, as found in the `LICENSE` file.
