# -*- coding: utf-8 -*-

import sys
import os
import numpy as np

import h5py
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import paddle
import paddle.fluid as fluid

# 读取数据部分----------------------------

# 用于存储信息
trainset_org = []
trainlabel_org = []
testset_org = []
testlabel_org = []

def getData(str,x,y,val):
	cnt = 0 # 明白了这里之所以会killed掉的原因就是数据量太大算力不够，姑且弄成100
	for filename in os.listdir(str):
		x.append(cv2.imread(str + '/' + filename))
		y.append(val)
		cnt = cnt + 1
		if cnt >= 100:
			break

# 读取所有的数据包括训练集和测试集
getData("../train/R0",trainset_org,trainlabel_org,0)
getData("../train/B1",trainset_org,trainlabel_org,1)
getData("../train/M2",trainset_org,trainlabel_org,2)
getData("../train/S3",trainset_org,trainlabel_org,3)
getData("../test/R0",testset_org,testlabel_org,0)
getData("../test/B1",testset_org,testlabel_org,1)
getData("../test/M2",testset_org,testlabel_org,2)
getData("../test/S3",testset_org,testlabel_org,3)

# 先对图片数据进行处理
trainset_org = np.array(trainset_org)
m_train = trainset_org.shape[0]

testset_org = np.array(testset_org) 
m_test = testset_org.shape[0]
num_px = testset_org.shape[1]

DATA_DIM = num_px * num_px * 3 # 定义维度

trainset_flatten = trainset_org.reshape(m_train,-1) # 训练集降维
testset_flatten = testset_org.reshape(m_test,-1) # 测试集降维
trainset_x = trainset_flatten/255. # 训练集归一化
testset_x = testset_flatten/255. # 测试集归一化

# 再对label进行处理
trainlabel_org = np.array(trainlabel_org)
trainset_y = np.array([trainlabel_org])

testlabel_org = np.array(testlabel_org)
testset_y = np.array([testlabel_org])

# 接下来看数组横向合并，如果出问题可能是对label的处理不对，我这个没省略号。。
train_set = np.hstack((trainset_x,trainset_y.T))
test_set = np.hstack((testset_x,testset_y.T))

# 至此基本材料都应该准备好了

def read
