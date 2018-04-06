from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    with open('requirements.txt') as f:
        return f.read()

setup(
    name='fbtranslate',
    version='0.1',
    author='Facebook AI',
    description=('Facebook Translation System'),
    long_description=readme(),
    url='https://github.com/facebookincubator/fbtranslate',
    license='BSD',
    packages=find_packages(),
    install_requires=['fairseq'],
)
