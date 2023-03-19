#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/17 09:13
# @Author  : Rocky
# @Site    : 
# @File    : train.py
# @Software: PyCharm
import time
from cifar10_test.model import model
import paddle
import paddle.nn  as nn
import paddle.nn.functional as F
from cifar10_test.dataset import trainloader, testloader, BATCH_SIZE
from time import strftime
from time import gmtime

epoch_num = 10
LR = 0.001
optimizer = paddle.optimizer.Adam(learning_rate=LR, parameters=model.parameters())
loss_function = nn.CrossEntropyLoss()
# 在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
def train(model):
    model.train()
    for epoch in range(epoch_num):
        time_start = time.time()
        for batch_id,data in enumerate(trainloader()):
            x_data = paddle.cast(data[0], 'float32')
            y_data = paddle.cast(data[1], 'int32')
            y_data = paddle.reshape(y_data,(-1,1)) # 丢失维度信息需要增加维度
            y_predict = model(x_data)
            loss = F.cross_entropy(y_predict,y_data)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            print("训练轮次: {}; 损失: {}".format(epoch, loss.numpy()))

    # 保存模型参数
    paddle.save(model.state_dict(), '../cifar10_test/cifar_10.pdparams')




if __name__ == '__main__':
    train(model)
