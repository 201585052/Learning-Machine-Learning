# !ls /home/aistudio/data/data851

# !ls /home/aistudio/work/

# # -*- coding: utf-8 -*-

# import numpy as np
# import paddle
# import paddle.fluid as fluid
# import matplotlib.pyplot as plt
# import pandas as pd
# %matplotlib inline

# colnames = ['性别']+['长度']+['直径']+['高度']+['总重量']+['皮重']+['内脏重量']+['壳重']+['年龄']
# # male是1，female是-1，I是0
# print_data = pd.read_csv('/home/aistudio/data/data851/baoyu.txt',names = colnames,sep = '\t')
# print_data.head()

# global x_raw,train_data,test_data
# data = np.loadtxt('/home/aistudio/data/data851/baoyu.txt',delimiter = '\t')
# x_raw = data.copy() 
# print x_raw

# #axis=0,表示按列计算
# #data.shape[0]表示data中一共有多少列
# maximums,minimums,avgs = data.max(axis=0),data.min(axis=0),data.sum(axis=0)/data.shape[0]
# # 和房价不同的是这里找的是因变量向量
# print "the raw area :",data[:,:8].max(axis = 0)

# #归一化，data[:,i]表示第i列的元素

# ### START CODE HERE ### (≈ 3 lines of code)
# feature_num = 9
# for i in xrange(feature_num-1):
#     data[:,i] = (data[:,i]-np.mean(data[:,i]))/(data[:,i].max(axis = 0) - data[:,i].min(axis = 0))
# ### END CODE HERE ###

# print 'normalization:',data[:,:8].max(axis = 0)

# ratio = 0.8
# offset = int(data.shape[0]*ratio)

# ### START CODE HERE ### (≈ 2 lines of code)
# train_data = data[:offset].copy()
# test_data = data[offset:].copy()
# ### END CODE HERE ###

# print(len(data))
# print(len(train_data))

# def read_data(data_set):
#     """
#     一个reader
#     Args：
#         data_set -- 要获取的数据集
#     Return：
#         reader -- 用于获取训练集及其标签的生成器generator
#     """
#     def reader():
#         """
#         一个reader
#         Args：
#         Return：
#             data[:-1],data[-1:] --使用yield返回生成器
#                 data[:-1]表示前n-1个元素，也就是训练数据，
#                 data[-1:]表示最后一个元素，也就是对应的标签
#         """
#         ### START CODE HERE ### (≈ 2 lines of code)
#         for i in range(len(data_set)):
#             yield data_set[i][:-1],data_set[i][-1:]
#         ### END CODE HERE ###
#     return reader

# def train():
#     """
#     定义一个reader来获取训练数据集及其标签
#     Args：
#     Return：
#         read_data -- 用于获取训练数据集及其标签的reader
#     """
#     global train_data
#     return read_data(train_data)

# def test():
#     global test_data
#     return read_data(test_data)
# #使用CPU训练
# use_cuda = False
# place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# def train_program():
#     ### START CODE HERE ### (≈ 5 lines of code)
#     y = fluid.layers.data(name='y', shape=[1], dtype='float32')

#     # feature vector of length 1
#     x = fluid.layers.data(name='x', shape=[8], dtype='float32')
#     y_predict = fluid.layers.fc(input=x, size=1, act=None)

#     loss = fluid.layers.square_error_cost(input=y_predict, label=y)
#     avg_loss = fluid.layers.mean(loss)
    
#     ### END CODE HERE ###

#     return avg_loss

# # 此处出现第一个可调参数learning_rate
# # 创建optimizer，更多优化算子可以参考 fluid.optimizer()
# def optimizer_program():
#     ### START CODE HERE ### (≈ 1 lines of code)
#     return fluid.optimizer.SGD(learning_rate= 0.01 )
#     ### END CODE HERE ###

# # 个人认为这一部分的可调参数主要体现在图像的绘制上，对实际分析的结果影响不大
# # 数据层和数组索引映射，用于trainer训练时喂数据
# feed_order=['x', 'y']
# # 保存模型
# params_dirname = "easy_fit_a_line.inference.model"

# # Plot data
# from paddle.v2.plot import Ploter
# train_title = "Train cost"
# test_title = "Test cost"
# plot_cost = Ploter(train_title, test_title)

# step = 0

# # 事件处理
# def event_handler_plot(event):
#     global step
#     if isinstance(event, fluid.EndStepEvent):
#         if event.step % 10 == 0:
#             plot_cost.append(train_title, step, event.metrics[0])
#             plot_cost.plot()
#         if event.step % 100 == 0: # 每10个batch,记录cost
#             test_metrics = trainer.test(
#             reader=test_reader, feed_order=feed_order)

#             plot_cost.append(test_title, step, test_metrics[0])
#             plot_cost.plot()

#             if test_metrics[0] < 8.0:
#                 # 如果准确率达到阈值，则停止训练
#                 print('loss is less than 8.0, stop')
#                 trainer.stop()

#         # 将参数存储，用于预测使用
#         if params_dirname is not None:
#             trainer.save_params(params_dirname)

#     step += 1

# # 创建执行器，palce在程序初始化时设定
# exe = fluid.Executor(place)
# # 初始化执行器
# exe.run(fluid.default_startup_program())

# BATCH_SIZE = 24

# # 设置训练reader
# train_reader = paddle.batch(
# paddle.reader.shuffle(
#     train(), buf_size=500),
#     batch_size=BATCH_SIZE)

# #设置测试reader
# test_reader = paddle.batch(
# paddle.reader.shuffle(
#     test(), buf_size=500),
# batch_size=BATCH_SIZE)

# trainer = fluid.Trainer(
#     train_func=train_program,
#     place=place,
#     optimizer_func=optimizer_program)

# print(fluid.default_main_program().to_string(True))

# trainer.train(
#     reader=train_reader,
#     ### START CODE HERE ### (≈ 1 lines of code)
#     num_epochs=300,
#     ### END CODE HERE ###
#     event_handler=event_handler_plot,
#     feed_order=feed_order)

# # 接下来就是预测程序了
# def inference_program():
#     x = fluid.layers.data(name='x', shape=[8], dtype='float32')
#     y_predict = fluid.layers.fc(input=x, size=1, act=None)
#     return y_predict

# inferencer = fluid.Inferencer(
#     infer_func=inference_program, param_path=params_dirname, place=place)

# batch_size = 8
# tensor_x = np.random.uniform(0, 1, [batch_size, 8]).astype("float32")

# results = inferencer.infer({'x': tensor_x})
# raw_x = tensor_x*(maximums[i]-minimums[i])+avgs[i]
# print("the area is:",raw_x)
# print("infer results: ", results[0])

# # 多元模型a、b、c、d、e、f、g、h参数,统一放到weight数组里作为权重向量
# ### START CODE HERE ### (≈ 2 lines of code)
# weight= np.linalg.solve(raw_x,results[0][:,0])
# ### END CODE HERE ###
# print weight

# # 和房价那个不同，因为这里有多维因变量，所以只能通过损失估计的方法来衡量
# data = np.loadtxt('/home/aistudio/data/data851/baoyu.txt',delimiter = '\t')
# x = data[:,:8]
# y = data[:,-1]
# y_predict_before = weight * x
# y_predict = []
# for i in y_predict_before:
#     y_predict.append(sum(i))
# def rssError(yArr, yHatArr):
#     """
#     误差大小评价函数
#     Parameters:
#         yArr - 真实数据
#         yHatArr - 预测数据
#     Returns:
#         误差大小
#     """
#     return ((yArr - yHatArr) **2).sum()

# print rssError(y,y_predict)


