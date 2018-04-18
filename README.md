# fbtranslate

## Installation instructions

### Install MiniConda3
```bash
pushd ~/local && \
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p ~/local/miniconda && \
    rm miniconda.sh
popd
```

### Install PyTorch
```bash
. ~/local/miniconda/bin/activate
conda uninstall -y pytorch pytorch-nightly
conda install -y pytorch-nightly -c pytorch
```

### Install FBTranslate
```bash
. ~/local/miniconda/bin/activate
yes | pip uninstall fbtranslate
git clone --recursive https://github.com/facebookincubator/fbtranslate.git
cd fbtranslate
python3 setup.py build develop
```
