from setuptools import setup, find_packages, Extension

def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    with open('requirements.txt') as f:
        return f.read()

bleu = Extension(
    'fairseq.libbleu',
    sources=[
        'fairseq/fairseq/clib/libbleu/libbleu.cpp',
        'fairseq/fairseq/clib/libbleu/module.cpp',
    ],
    extra_compile_args=['-std=c++11'],
)

setup(
    name='fbtranslate',
    version='0.1',
    author='Facebook AI',
    description=('Facebook Translation System'),
    long_description=readme(),
    url='https://github.com/facebookincubator/fbtranslate',
    license='BSD',
    packages=find_packages(),
    ext_modules=[bleu],
    install_requires=['fairseq'],
)
