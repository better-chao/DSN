import numpy as np
import hickle as hkl
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import urllib
import os
import tarfile
# import skimage
# import skimage.io
# import skimage.transform
import torch
from PIL import Image

# 这个是打乱数据的操作
def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""

    num = data[0].shape[0]
    p = np.random.permutation(num)

    return [d[p] for d in data]

# 这个是生成批量化数据
def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                # print("此处调用了我")
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        # print("norm_transform:", type(norm_transform[0]))
        # norm_transform: <class 'torchvision.transforms.transforms.Normalize'>
        # norm_transform: [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(transform_train) or img_.max() < 1:
        img_ = img_.detach().numpy() * 255
    
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )

    return img_

def show(batch_source, batch_target, batch_y_s, batch_y_t, train_transform):
    # 该函数的主要作用是将取出来的图像进行显示，尤其是观察目标域的图像，这里主要是观察图像的标签和图像内容是不是对得上
    number = batch_source.shape[0]
    print("number:", number, type(number))
    batch_y_s = torch.argmax(batch_y_s, dim=1)
    for i in range(number):
        image = transform_invert(batch_source[i], train_transform)
        plt.imshow(image)
        plt.text(-0.5, -0.5, str(batch_y_s[i].item()), fontsize=20, color='black')
        plt.savefig('./figs/source/source_{}.jpg'.format(i))
        plt.close()
    batch_y_t = torch.argmax(batch_y_t, dim=1)
    for i in range(number):
        image = transform_invert(batch_target[i], train_transform)
        plt.imshow(image)
        plt.text(-0.5, -0.5, str(batch_y_t[i].item()), fontsize=20, color='black')
        plt.savefig('./figs/target/target_{}.jpg'.format(i))
        plt.close()