# -*- coding: utf-8 -*-

import sys
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as pimg
from PIL import Image
import paddle
import paddle.fluid as fluid
from paddle.v2.plot import Ploter

# 读取数据部分----------------------------

# 用于存储信息
trainset_org = []
trainlabel_org = []
testset_org = []
testlabel_org = []

def cutImage(img):
    width,height = img.size
    return img.resize(((int)(0.4 * width),(int)(0.4 * height)))
def getData(str,x,y,val):
    for filename in os.listdir(str):
        x.append(np.array(cutImage(Image.open(str + '/' + filename))).flatten()) # 不是数据算力的问题 可以把图片变小
        y.append(np.array(val))


# 读取所有的数据包括训练集和测试集
getData("/home/aistudio/petch/peech/train/R0",trainset_org,trainlabel_org,0)
getData("/home/aistudio/petch/peech/train/B1",trainset_org,trainlabel_org,1)
getData("/home/aistudio/petch/peech/train/M2",trainset_org,trainlabel_org,2)
getData("/home/aistudio/petch/peech/train/S3",trainset_org,trainlabel_org,3)
getData("/home/aistudio/petch/peech/test/R0",testset_org,testlabel_org,0)
getData("/home/aistudio/petch/peech/test/B1",testset_org,testlabel_org,1)
getData("/home/aistudio/petch/peech/test/M2",testset_org,testlabel_org,2)
getData("/home/aistudio/petch/peech/test/S3",testset_org,testlabel_org,3)

# trainset的左右侧
trainset_org = np.array(trainset_org)
trainlabel_org = np.array(trainlabel_org).reshape(-1,1)

# testset的左右侧
testset_org = np.array(testset_org)
testlabel_org = np.array(testlabel_org).reshape(-1,1)

# 归一化
trainset_org = trainset_org/255.
testset_org = testset_org/255.

# print trainset_org

train_set = np.hstack((trainset_org, trainlabel_org))
test_set = np.hstack((testset_org, testlabel_org))

DATA_DIM = train_set.shape[1] - 1
print DATA_DIM

train_set = train_set.reshape(-1,train_set.shape[1])
np.random.shuffle(train_set) # 随机分布很重要
test_set = test_set.reshape(-1,test_set.shape[1])
np.random.shuffle(test_set) # 同上
# 至此唯一的问题好像是np.array的问题
print train_set.shape
print train_set

# 定义reader

# 读取训练数据或测试数据
def read_data(data_set):
    """
        一个reader
        Args:
            data_set -- 要获取的数据集
        Return:
            reader -- 用于获取训练数据集及其标签的生成器generator
    """
    def reader():
        """
        一个reader
        Args:
        Return:
            data[:-1], data[-1:] -- 使用yield返回生成器(generator)，
                    data[:-1]表示前n-1个元素，也就是训练数据，data[-1:]表示最后一个元素，也就是对应的标签
        """
        for data in data_set:
            yield data[:-1], data[-1]
    return reader

# 开始训练

# 使用CPU训练
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# 配置网络结构和设置参数

def infer_func():
    x = fluid.layers.data(name='x', shape=[DATA_DIM], dtype='float32')
    h1 = fluid.layers.fc(input=x, size=16, act='relu')
    h2 = fluid.layers.fc(input=h1,size=32, act='relu')
    y_predict = fluid.layers.fc(input=h2, size=4, act='softmax')
    return y_predict

feeder = None

#输入层与标签层作为输入传进函数，这里只需定义输出层与损失函数                          
def train_func():
    global feeder
    y_predict = infer_func()
    y = fluid.layers.data(name='y', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    feeder = fluid.DataFeeder(place=place, feed_list=['x', 'y'])
    return [avg_cost, y_predict]

# 这里定义学习率
def optimizer_func():
    return fluid.optimizer.Adam(learning_rate=0.0001)

# 创建训练器

trainer = fluid.Trainer(
    train_func= train_func,
    place= place,
    optimizer_func= optimizer_func)

feed_order=['x','y']

BATCH_SIZE=40

# 设置训练reader
train_reader = paddle.batch(
    paddle.reader.shuffle(
        read_data(train_set), buf_size=500),
    batch_size=BATCH_SIZE)

#设置测试 reader
test_reader = paddle.batch(
    paddle.reader.shuffle(
        read_data(test_set), buf_size=500),
    batch_size=BATCH_SIZE)

save_dirname="recognize_peech_inference.model"

# 定义回调函数

# Plot data

train_title = "Train cost"
test_title = "Test cost"
plot_cost = Ploter(train_title, test_title)

step = 0
# 事件处理
def event_handler_plot(event):
    global step
    if isinstance(event, fluid.EndStepEvent):
        if event.step % 2 == 0: # 若干个batch,记录cost
            if event.metrics[0] < 10:
                plot_cost.append(train_title, step, event.metrics[0])
                plot_cost.plot()
        if event.step % 20 == 0: # 若干个batch,记录cost
            test_metrics = trainer.test(
                reader=test_reader, feed_order=feed_order)
            if test_metrics[0] < 10:
                plot_cost.append(test_title, step, test_metrics[0])
                plot_cost.plot()

#             if test_metrics[0] < 1.0:
#                 # 如果准确率达到阈值，则停止训练
#                 print('loss is less than 10.0, stop')
#                 trainer.stop()

   # 将参数存储，用于预测使用
        if save_dirname is not None:
            trainer.save_params(save_dirname)
    step += 1
   # plot_cost.savefig("./planecost.jpg")
   # 考虑一下图片的实时绘制和保存问题

# 开始训练了
EPOCH_NUM = 10 # 整体训练轮数
trainer.train(
    reader=train_reader,
    num_epochs=EPOCH_NUM,
    event_handler=event_handler_plot,
    feed_order=feed_order)

inferencer = fluid.Inferencer(
    infer_func=infer_func, param_path=save_dirname, place=place)

BATCH_SIZE = 10
test_reader = paddle.batch(
    read_data(test_set), batch_size=BATCH_SIZE
)

# 取出一个 mini-batch
for mini_batch in test_reader(): 
    # 转化为 numpy 的 ndarray 结构，并且设置数据类型
    test_x = np.array([data[0] for data in mini_batch]).astype("float32")
    test_y = np.array([data[1] for data in mini_batch]).astype("int64")
    # 真实进行预测
    mini_batch_result = inferencer.infer({'x': test_x})
    
    # 打印预测结果
    mini_batch_result = np.argsort(mini_batch_result) #找出可能性最大的列标，升序排列
    mini_batch_result = mini_batch_result[0][:, -1]  #把这些列标拿出来
    print('预测结果：%s'%mini_batch_result)
    
    # 打印真实结果    
    label = np.array(test_y) # 转化为 label
    print('真实结果：%s'%label)
    break

# 查看百分比
def right_ratio(right_counter, total):
    ratio = float(right_counter)/total
    return ratio

# 评估函数 data_set 是一个reader
def evl(data_set):
    total = 0    #操作的元素的总数
    right_counter = 0  #正确的元素

    pass_num = 0
    for mini_batch in data_set():
        pass_num += 1
        #预测
        test_x = np.array([data[0] for data in mini_batch]).astype("float32")
        test_y = np.array([data[1] for data in mini_batch]).astype("int64")
        mini_batch_result = inferencer.infer({'x': test_x})
        
        #预测的结果
        mini_batch_result = np.argsort(mini_batch_result) #找出可能性最大的列标，升序排列
        mini_batch_result = mini_batch_result[0][:, -1]  #把这些列标拿出来
#         print('预测结果：%s'%mini_batch_result)

        label = np.array(test_y) # 转化为 label
#         print('真实结果：%s'%label)

        #计数
        label_len = len(label)
        total += label_len
        for i in xrange(label_len):
            if mini_batch_result[i] == label[i]:
                right_counter += 1

    ratio = right_ratio(right_counter, total)
    return ratio

ratio = evl(train_reader)
print('训练数据的正确率 %0.2f%%'%(ratio*100))

ratio = evl(test_reader)
print('预测数据的正确率 %0.2f%%'%(ratio*100))