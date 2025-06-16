## This is a pytorch implementation of the [Domain Separation Networks](https://arxiv.org/abs/1608.06019)

## Environment
- numpy                1.21.5
- protobuf             3.20.0
- scikit-image         0.19.3
- tensorboard          1.15.0
- tensorflow           1.15.0
- tensorflow-estimator 1.15.1
- termcolor            2.3.0
- torch                1.12.1
- torchvision          0.13.1

## Network Structure

![model](./extra/model.jpg)

## Usage

`python model.py`

**Note that this model is very sensitive to the loss weight, our implementation cannot perform as perfect as the
original paper, so be careful when you tune parameters for other datasets. Moreover, this model may not be suitable
for real nature image, cause the private and shared feature of nature image are more complicated, so that *difference 
loss* cannot adapt well** 

## Result

We only conduct the experiments from mnist to mnist_m, the target accuracy of our implementation is about 86% (original
paper ~83%), and some results are shown as follows, from left to right: recovery from shared+private, shared and private
features.
