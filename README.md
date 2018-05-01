# Translate - a PyTorch Language Library

Translate is a library for machine translation written in PyTorch. It provides training for sequence-to-sequence models. These models can be exported to Caffe2 graphs via [ONNX](https://onnx.ai/), loaded and run from C++ for production purposes. Translate relies on [fairseq](https://github.com/facebookresearch/fairseq-py), a general sequence-to-sequence library.

## Requirements and Installation

Translate requires
* Mac OS X or Linux
* A [CUDA installation](https://docs.nvidia.com/cuda/)

To install Translate, please refer to the `install.sh` script. In short, run `bash install.sh`.

## Training

We provide a example script to train a model for the IWSLT 2014 German-English task. We used this command to obtain [a pretrained model](https://download.pytorch.org/models/translate/iwslt14/model.tar.gz):

```
bash translate/examples/train_iwslt14.sh
```

The pretrained model actually contains two checkpoints that correspond to training twice with random initialization of the parameters. This is useful to obtain ensembles.

## Pretrained Model

A pretrained model for IWSLT 2014 can be evaluated by running the example script:

```
bash translate/examples/generate_iwslt14.sh
```

Note the improvement in performance when using an ensemble of size 2 instead of a single model.

## Exporting a Model with ONNX

We provide an example script to export a PyTorch model to a Caffe2 graph via ONNX:

```
bash fbtranslate/examples/export_iwslt14.sh
```

TODO: add how to load the exported models from C++.

## Join the Translate community

We welcome contributions! See the `CONTRIBUTING.md` file for how to help out.

## License
Translate is BSD-licensed, as found in the `LICENSE` file.
