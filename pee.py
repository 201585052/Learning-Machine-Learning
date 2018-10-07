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

# 数据读取

trainset_org = []
trainlabel_org = []
for filename in os.listdir(r"../train/R0"):
    trainset_org.append(cv2.imread("../train/R0/"+filename))
    trainlabel_org.append([0])
for filename in os.listdir(r"../train/B1"):
	trainset_org.append(cv2.imread("../train/B1/"+filename))
	trainlabel_org.append([1])
for filename in os.listdir(r"../train/M2"):
	trainset_org.append(cv2.imread("../train/M2/"+filename))
	trainlabel_org.append([2])
for filename in os.listdir(r"../train/S3"):
	trainset_org.append(cv2.imread("../train/S3/"+filename))
	trainlabel_org.append([3])

trainset = np.array(trainset_org) # trainset代表处理后的64*64*3的所有图片集合
trainlabel_org = np.array(trainlabel_org[:])
trainlabel = trainlabel_org.reshape((1,trainlabel_org.shape[0]))# trainlabel代表所有图片对应的桃子大小分别为0,1,2,3



