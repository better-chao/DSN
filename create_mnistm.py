import tarfile
import os
import hickle as hkl
import numpy as np
import skimage
import gzip
from tensorflow.examples.tutorials.mnist import input_data

def compose_image(digit, background):
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)
    bg = background[x:x + dw, y:y + dh]
    return np.abs(bg - digit).astype(np.uint8)

def mnist_to_img(x):
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)

def create_mnistm(X, background_data):
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):
        index = rand.randint(0, len(background_data))
        bg_img = background_data[index]
        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img)
        X_[i] = d
    return X_

def convert_to_one_hot(labels, num_classes=10):
    """将整数标签转换为one-hot编码"""
    return np.eye(num_classes)[labels]

def savefile(history, path):
    hkl.dump(history, path)
    print("{} save success!".format(path))

if __name__ == '__main__': 

    # Save the MNIST images and labels, as well as the MNIST-M images, all in HKL format.
    data_directory = 'data/' 
    os.makedirs(data_directory, exist_ok=True)  # 如果目录已存在，不会报错  

    # 设置随机种子
    rand = np.random.RandomState(42)
    # 处理mnsits数据集
    mnist        = input_data.read_data_sets('MNIST_data', one_hot=True)
    # 这个是实现将单通道数据转为三通道数据
    mnist_train  = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
    mnist_train  = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test   = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
    mnist_test   = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    mnist_train_label = mnist.train.labels
    mnist_test_label = mnist.test.labels

    # save minst image as pickle
    mnist_data={'train':mnist_train,'test':mnist_test}
    path = os.path.join(str(data_directory), 'mnist_data.hkl')
    savefile(mnist_data, path)

    # save minst label as pickle
    mnist_label_data={'train_label':mnist_train_label,'test_label':mnist_test_label}
    path = os.path.join(str(data_directory), 'mnist_data_label.hkl')
    savefile(mnist_label_data, path)

    # To extract color backgrounds from images in BSR_bsds500.tgz and store them in the background_data list,
    BST_PATH = 'BSR_bsds500.tgz'
    f = tarfile.open(BST_PATH)
    train_files = []
    for name in f.getnames():
        if name.startswith('BSR/BSDS500/data/images/train/'):
            train_files.append(name)

    background_data = []
    for name in train_files:
        try:
            fp = f.extractfile(name)
            bg_img = skimage.io.imread(fp)
            background_data.append(bg_img)
        except:
            continue

    # produce minstm images
    minstm_train = create_mnistm(mnist.train.images, background_data)
    minstm_test = create_mnistm(mnist.test.images, background_data)
    minstm_valid = create_mnistm(mnist.validation.images, background_data)

    # save minstm images, the labels of minstm shared with minst
    mnistm_data = {'train':minstm_train,'test':minstm_test,'valid':minstm_valid}
    path = os.path.join(str(data_directory), 'mnistm_data.hkl')
    savefile(mnistm_data, path)

