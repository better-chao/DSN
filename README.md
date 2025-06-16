<div align="center">
<h1>[NeurIPS2016] Domain Separation Network</h1>
</div>

<div align="center">
  <img src="extra/model.jpg" width="800"/>
</div><br/>

## Notes
This is a Pytorch implementation of the Domain Separation Network [Arxiv](https://arxiv.org/abs/1608.06019). There have been multiple implementations of this article on GitHub, but most of them are based on TensorFlow with poor readability. The few PyTorch-based implementations have extremely brief documentation, making it difficult to start training and reproduce the metrics in the paper. This project provides a PyTorch implementation with clear steps, facilitating everyone to reproduce the results.

## Get Started

#### 1. Please follow these steps to create environment.

a. Create a conda virtual environment and activate it.

```shell
conda create -n dsn python=3.7 -y
conda activate dsn
```
b. Install other packets as followings.

- numpy                1.21.5
- protobuf             3.20.0
- scikit-image         0.19.3
- tensorboard          1.15.0
- tensorflow           1.15.0
- tensorflow-estimator 1.15.1
- torch                1.12.1
- torchvision          0.13.1

#### 2. Download [BSR_bsds500.tgz](https://drive.google.com/file/d/1gSUgdH1MdPZjGreUa8COnuem5pUTp8iA/view?usp=drive_link) and place it in the main directory.

#### 3. Generate training and testing datasets.

```shell
python create_mnistm.py
```
The script will automatically create a data directory and generate mnist_data_label.hkl, mnist_data.hkl, and mnistm_data.hkl in this directory for training and testing.

data/

├── mnist_data_label.hkl        # MNIST labels（one-hot, shared with MNIST-M）

├── mnist_data.hkl              # MNIST images（training + test）

├── mnistm_data.hkl             # MNIST-M images（traning + test + validation）

#### 4. Start Training and get the [trainingg.log](trainingg.log).

```shell
python model.py
```

**Note that this model is very sensitive to the batch_size, our implementation cannot perform as perfect as the
original paper, so be careful when you tune parameters for other datasets.** 

## Result

We only conduct the experiments from mnist to mnist_m, the target accuracy of our implementation is about 86% (original
paper ~83%).
