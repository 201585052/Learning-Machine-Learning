# -*- coding: utf-8 -*-

import sys
import numpy as np

import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import paddle
import paddle.fluid as fluid


# 数据读取

train_dataset = h5py.File('data/train_planevnonplane.h5', "r")
train_keys = train_dataset.keys()

train_set_x_orig = np.array(train_dataset[train_keys[0]][:]) # train set
train_set_y_orig = np.array(train_dataset[train_keys[1]][:]) # train label
train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))


test_dataset = h5py.File('data/test_planevnonplane.h5', "r")
test_keys  =test_dataset.keys()

test_set_x_orig = np.array(test_dataset[test_keys[0]][:]) # test set
test_set_y_orig = np.array(test_dataset[test_keys[1]][:]) # test label
test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

# 中间这一部分是一些简单的测试

index = 10
plt.imshow(train_set_x_orig[index])

# print train_set_y[:,index]

# 数据预处理

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]


# print ("训练样本数: m_train = " + str(m_train))
# print ("测试样本数: m_test = " + str(m_test))
# print ("图片高度/宽度: num_px = " + str(num_px))
# print ("图片大小: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_set_x shape: " + str(train_set_x_orig.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x shape: " + str(test_set_x_orig.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))rint ("训练样本数: m_train = " + str(m_train))

# 定义维度
DATA_DIM = num_px * num_px * 3

# 转换数据形状

train_set_x_flatten = train_set_x_orig.reshape(m_train,-1)
test_set_x_flatten = test_set_x_orig.reshape(m_test,-1)

# print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))

# 归一化

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

# numpy 数组的横向合并

print train_set_x_orig
train_set = np.hstack((train_set_x, train_set_y.T))
test_set = np.hstack((test_set_x, test_set_y.T))

