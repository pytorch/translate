# translate

## Installation instructions

### Install MiniConda3
```bash
pushd ~
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p ~/miniconda
    rm miniconda.sh
popd
```

### Install PyTorch
```bash
. ~/miniconda/bin/activate
conda uninstall -y pytorch pytorch-nightly
conda install -y pytorch-nightly -c pytorch
```

### Install FBTranslate
```bash
. ~/miniconda/bin/activate
yes | pip uninstall translate
git clone --recursive https://github.com/facebookincubator/translate.git
cd translate
python3 setup.py build develop
```
