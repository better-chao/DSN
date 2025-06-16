# coding: utf-8
# 本文档用于自己复现domain separation network这篇文章
# 这是自己第一次复现文章，采用的框架是pytorch，在复现过程中需要注意的几个点是：
# 1.了解pytorch搭建整个神经网络的流程和一般训练方式
# 2.测试pytorch固定种子的做法
# 3.测试多卡脚本的做法

# 在复现该文章的同时，要形成自己的一些敲代码的规范
# pytorch 编写神经网络的代码分为以下数据、模型、损失函数、优化器、迭代训练
# 本部分我们来完成数据部分的编写
# 本次复现的部分仅仅是用到mnist与mnist-m数据集
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import numpy as np
import hickle as hkl
import os
import random
from torch.utils.data import Dataset
import copy

# # 封装一个读取数据的函数
# def read_data():
#     # 读取mnist数据
#     mnist_name  = './data/mnist_data.hkl'
#     mnist      = hkl.load(open(mnist_name))
#     mnist_train = mnist["train"]
#     mnist_test = mnist["test"]
#     # print("mnist.train", mnist["train"].shape)  # mnistm.train (55000, 28, 28, 3)
#     # print("mnist.test", mnist["test"].shape)  # mnistm.test (10000, 28, 28, 3)
#     # print("mnist.train.type", type(mnist["train"]))
#     # 读取mnist标签数据
#     mnist_label_name  = './data/mnist_data_label.hkl'
#     mnist_label      = hkl.load(open(mnist_label_name))
#     mnist_train_label =  mnist_label["train_label"]
#     mnist_test_label = mnist_label["test_label"]
#     # print("mnist_train_label", mnist_label["train_label"].shape)  # mnistm.train (55000, 28, 28, 3)
#     # print("mnist_test_label", mnist_label["test_label"].shape)  # mnistm.test (10000, 28, 28, 3)
#     # 读取mnist-M数据

#     mnistm_name  = './data/mnistm_data.hkl'
#     mnistm       = hkl.load(open(mnistm_name))
#     mnistm_train = mnistm['train']
#     mnistm_test  = mnistm['test']
#     mnistm_train_label = mnist_train_label
#     mnistm_test_label = mnist_test_label
#     # print("mnistm.train", mnistm["train"].shape)  # mnistm.train (55000, 28, 28, 3)
#     # print("mnistm.test", mnistm["test"].shape)  # mnistm.test (10000, 28, 28, 3)

#     # 计算一个pixel_mean
#     pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

#     return mnist_train, mnist_test, mnist_train_label, mnist_test_label,\
#          mnistm_train, mnistm_test, mnistm_train_label, mnistm_test_label, pixel_mean

# 根据这个预处理的手段，我们在前需要计算整个数据集的均值
def calc_mean(train_dir):
    mnist_name  = './data/mnist_data.hkl'
    mnist      = hkl.load(open(mnist_name))
    mnist_train = mnist["train"]

    mnistm_name  = './data/mnistm_data.hkl'
    mnistm       = hkl.load(open(mnistm_name))
    mnistm_train = mnistm['train']

    pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))
    # print("pixel_mean:", pixel_mean)
    pixel_mean = pixel_mean / 255.0
    # print("pixel_mean:", pixel_mean)

    return pixel_mean

# # 在这里封装一个数据集类,保证读数据之后能够返回一个batch的mnist和mnistm的数据
# class Mnist_and_MnistM_Dataset(Dataset):
#     def __init__(self, data_dir, transform=None, mode ='train'):
#         """
#         rmb面额分类任务的Dataset
#         :param data_dir: str, 数据集所在路径
#         :param transform: torch.transform，数据预处理
#         """
#         self.mode = mode
#         self.mnist_name = os.path.join(data_dir, 'mnist_data.hkl') 
#         self.Mnist_img = hkl.load(open(self.mnist_name))['train']
#         self.Mnist_img_test = hkl.load(open(self.mnist_name))['test']
#         self.shuffle_ix = np.random.permutation(np.arange(self.__len__()))
#         self.mnistm_name  = os.path.join(data_dir, 'mnistm_data.hkl')
#         if self.mode == 'train':
#             self.MnistM_img = hkl.load(open(self.mnistm_name))['train'][self.shuffle_ix, :]
#         else:
#             self.MnistM_img_test = hkl.load(open(self.mnistm_name))['test'][self.shuffle_ix, :]
#         self.mnist_label_name = os.path.join(data_dir, 'mnist_data_label.hkl')
#         self.Mnist_img_label = hkl.load(open(self.mnist_label_name))['train_label']
#         self.Mnist_img_label_test = hkl.load(open(self.mnist_label_name))['test_label']
#         self.MnistM_img_label = self.Mnist_img_label.copy()
#         self.MnistM_img_label_test = self.Mnist_img_label_test.copy()
#         if self.mode == 'train':
#             self.MnistM_img_label = self.MnistM_img_label[self.shuffle_ix, :]
#         else:
#             self.MnistM_img_label_test = self.MnistM_img_label_test[self.shuffle_ix, :]
#         self.transform = transform
        
#     def __getitem__(self, index):
#         if self.mode == "train":
#             source_batch = self.Mnist_img[index]
#             source_batch_y = self.Mnist_img_label[index]
#             # 接下来先将该数据进行打乱,再取索引
#             target_batch = self.MnistM_img[index]
#             target_batch_y = self.MnistM_img_label[index]

#             if self.transform is not None:
#                 Mnist_img = self.transform(source_batch)   # 在这里做transform，转为tensor等等
#                 MnistM_img = self.transform(target_batch)
#             Mnist_img_label = torch.tensor(source_batch_y)
#             MnistM_img_label = torch.tensor(target_batch_y)
#             return Mnist_img, MnistM_img, Mnist_img_label, MnistM_img_label
#         else:
#             source_batch = self.Mnist_img_test[index]
#             source_batch_y = self.Mnist_img_label_test[index]
#             # 接下来先将该数据进行打乱,再取索引
#             target_batch = self.MnistM_img_test[index]
#             target_batch_y = self.MnistM_img_label_test[index]

#             if self.transform is not None:
#                 Mnist_img = self.transform(source_batch)   # 在这里做transform，转为tensor等等
#                 MnistM_img = self.transform(target_batch)
#             Mnist_img_label = torch.tensor(source_batch_y)
#             MnistM_img_label = torch.tensor(target_batch_y)
#             return Mnist_img, MnistM_img, Mnist_img_label, MnistM_img_label

#     def __len__(self）
#         if self.mode == 'train':
#             return hkl.load(open(self.mnist_name))['train'].shape[0]
#         else:
#             return hkl.load(open(self.mnist_name))['test'].shape[0]

# 在这里将类的封装代码先改一下
# 在这里封装一个数据集类,保证读数据之后能够返回一个batch的mnist和mnistm的数据
class Mnist_and_MnistM_Dataset(Dataset):
    def __init__(self, data_dir, transform=None, mode ='train'):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.mode = mode
        self.mnist_name = os.path.join(data_dir, 'mnist_data.hkl') 
        self.Mnist_img = hkl.load(open(self.mnist_name))['train']
        self.Mnist_img_test = hkl.load(open(self.mnist_name))['test']
        
        self.mnistm_name  = os.path.join(data_dir, 'mnistm_data.hkl')
        if self.mode == 'train':
            self.MnistM_img = hkl.load(open(self.mnistm_name))['train']
        else:
            self.MnistM_img_test = hkl.load(open(self.mnistm_name))['test']
        self.mnist_label_name = os.path.join(data_dir, 'mnist_data_label.hkl')
        self.Mnist_img_label = hkl.load(open(self.mnist_label_name))['train_label']
        self.Mnist_img_label_test = hkl.load(open(self.mnist_label_name))['test_label']
        self.MnistM_img_label = self.Mnist_img_label.copy()
        self.MnistM_img_label_test = self.Mnist_img_label_test.copy()
        if self.mode == 'train':
            self.MnistM_img_label = self.MnistM_img_label
        else:
            self.MnistM_img_label_test = self.MnistM_img_label_test
        self.transform = transform
        
    def __getitem__(self, index):
        if self.mode == "train":
            source_batch = self.Mnist_img[index]
            source_batch_y = self.Mnist_img_label[index]
            # 接下来先将该数据进行打乱,再取索引
            target_batch = self.MnistM_img[index]
            target_batch_y = self.MnistM_img_label[index]

            if self.transform is not None:
                Mnist_img = self.transform(source_batch)   # 在这里做transform，转为tensor等等
                MnistM_img = self.transform(target_batch)
            Mnist_img_label = torch.tensor(source_batch_y)
            MnistM_img_label = torch.tensor(target_batch_y)
            return Mnist_img, MnistM_img, Mnist_img_label, MnistM_img_label
        else:
            source_batch = self.Mnist_img_test[index]
            source_batch_y = self.Mnist_img_label_test[index]
            # 接下来先将该数据进行打乱,再取索引
            target_batch = self.MnistM_img_test[index]
            target_batch_y = self.MnistM_img_label_test[index]

            if self.transform is not None:
                Mnist_img = self.transform(source_batch)   # 在这里做transform，转为tensor等等
                MnistM_img = self.transform(target_batch)
            Mnist_img_label = torch.tensor(source_batch_y)
            MnistM_img_label = torch.tensor(target_batch_y)
            return Mnist_img, MnistM_img, Mnist_img_label, MnistM_img_label

    def __len__(self):
        if self.mode == 'train':
            return hkl.load(open(self.mnist_name))['train'].shape[0]
        else:
            return hkl.load(open(self.mnist_name))['test'].shape[0]

if __name__ == '__main__': 
    result = calc_mean('./data')

    a = np.arange(30).reshape(5, 3, 2)
    print("a:", a, a.shape)
    print("a[1]:", a[1], a[1].shape)
    shuffle_ix = np.random.permutation(np.arange(5))
    b = a[shuffle_ix, :]
    print("b:", b, b.shape)
    print("b[1]:", b[1], b[1].shape)


