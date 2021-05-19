# !/bin/bash

mkdir data

pip3 install kaggle pandas

kaggle competitions download -c cifar-10
mv cifar-10.zip data
mkdir -p data/cifar-10/raw
unzip data/cifar-10.zip -d data/cifar-10/raw
mkdir -p data/cifar-10/prepared
7z e data/cifar-10/raw/train.7z -odata/cifar-10/raw/train
mkdir -p data/cifar-10/prepared/train data/cifar-10/prepared/test
python3 prepare_cifar.py

kaggle competitions download -c house-prices-advanced-regression-techniques
mv house-prices-advanced-regression-techniques.zip data
mkdir -p data/house-prices-advanced-regression-techniques/raw
unzip data/house-prices-advanced-regression-techniques.zip -d data/house-prices-advanced-regression-techniques/raw
mkdir -p data/house-prices-advanced-regression-techniques/prepared
