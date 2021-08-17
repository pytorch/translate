#!/usr/bin/env python3

from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


def requirements():
    with open("requirements.txt") as f:
        return f.read()


setup(
    name="pytorch-translate",
    version="0.1",
    author="Facebook AI",
    description=("Facebook Translation System"),
    long_description=readme(),
    url="https://github.com/pytorch/translate",
    license="BSD",
    packages=find_packages(),
    install_requires=[
        "fairseq>=0.5.0",
    ],
    dependency_links=[
        "git+https://github.com/pytorch/fairseq.git#egg=fairseq-0.5.0",
    ],
    test_suite="pytorch_translate",
)
