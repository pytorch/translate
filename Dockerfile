# Pre-req installations:
# https://docs.docker.com/install/linux/docker-ce/ubuntu/
# https://github.com/NVIDIA/nvidia-docker
# https://github.com/pytorch/pytorch/tree/master/docker/caffe2

# Usage: git clone --recursive https://github.com/pytorch/translate.git && sudo docker build -t pytorch_translate_initial_release . && sudo nvidia-docker run pytorch_translate_initial_release

FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install -y \
    python3.6 \
    python3-numpy \
    python3-setuptools \
    python-pip

    # Probably unneeded deps
    # python3-pydot \
    # python3-scipy \

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libprotobuf-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade setuptools wheel && \
    pip install --no-cache-dir \
    future \
    numpy \
    protobuf \
    pyyaml \
    requests \
    setuptools \
    six \
    tornado

    # Probably unneeded pip deps
    # graphviz \
    # hypothesis \
    # jupyter \
    # matplotlib \
    # flask \
    # python-nvd3 \
    # scikit-image \
    # scipy \

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p ${HOME}/miniconda && \
    rm miniconda.sh
# wget --no-check-certificate
RUN . ${HOME}/miniconda/bin/activate

ENV PATH ${HOME}/miniconda/bin:$PATH
ENV CONDA_PATH ${HOME}/miniconda

WORKDIR /translate
ADD translate /translate
ENV NCCL_ROOT_DIR /translate/nccl_2.1.15-1+cuda8.0_x86_64
ENV LD_LIBRARY_PATH ${CONDA_PATH}/lib:${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}
RUN echo $NCCL_ROOT_DIR
RUN echo $LD_LIBRARY_PATH

RUN ./install.sh
RUN ./pytorch_translate/examples/generate_iwslt14.sh
RUN ./pytorch_translate/examples/export_iwslt14.sh
RUN echo "hallo welt ." | ./pytorch_translate/examples/translate_iwslt14.sh
