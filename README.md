# Translate - a PyTorch Language Library

Translate is a library for machine translation written in PyTorch. It provides training for sequence-to-sequence models. Translate relies on [fairseq](https://github.com/pytorch/fairseq), a general sequence-to-sequence library, which means that models implemented in Translate and Fairseq can be trained. Translate also provides the ability to export some models to Caffe2 graphs via [ONNX](https://onnx.ai/) and to load and run these models from C++ for production purposes. Currently, we export components (encoder, decoder) to Caffe2 separatetly and beam search is implemented in C++. In the near future, we will be able to export the beam search as well. We also plan to add export support to more models.

## Requirements and Installation

Translate requires
* A Linux operating system with a CUDA compatible card
* C++ compiler supporting ECMAScript syntax for `<regex>`, such as GCC 4.9 and above.
* A [CUDA installation](https://docs.nvidia.com/cuda/). We recommend CUDA 8 or CUDA 9.

To install Translate, please refer to the `install.sh` script. In short, run `bash install.sh`. We have tested this script on CentOS 7.4.1708 with a Tesla M40 card and a CUDA 8 installation. We encourage you to report an [issue](https://github.com/pytorch/translate/issues) if you are unable to install this project for your specific configuration.

Alternatively, you can launch an AWS instance using the `pytorch_translate_tmp_1` image. Once you have ssh'ed to the instance, the example commands below should work after running `cd translate`.

## Training

Note: the example commands given assume that you are the root of the cloned gihub repository or that you're using an AWS instance and that you have run `cd translate`.

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

We're excited to welcome contributions! See the `CONTRIBUTING.md` file for how to help out.

## License
Translate is BSD-licensed, as found in the `LICENSE` file.
