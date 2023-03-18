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
    for epo in range(epoch_num):
        train_correct = 0
        train_sum = 0
        epoch_used_time = 0
        epoch_ave_time = 0
        time_start = time.time()
        for i, (img, label) in enumerate(trainloader):
            optimizer.clear_grad()
            model.train()
            out = model(img)

            prediction = paddle.argmax(out, 1)
            pre_num = prediction.cpu().numpy()
            train_correct += (pre_num == label.cpu().numpy()).sum()
            train_sum += BATCH_SIZE

            loss = loss_function(out, label)
            loss.backward()
            optimizer.step()

            epoch_used_time += (time.time() - time_start)
            time_start = time.time()

            # 加了个比较简陋的计时方式，显示训练剩余时间，以便估计摸鱼时间
            used_t = strftime("%H:%M:%S", gmtime(epoch_used_time))
            total_t = strftime("%H:%M:%S", gmtime((epoch_used_time / (i + 1)) * len(trainloader)))


            print(
                f"\rEpoch：{str(epo)} Iter {train_sum}/{len(trainloader) * BATCH_SIZE} Train ACC： {(train_correct / train_sum):.5}\
                Used_Time：{used_t} / Total_Time：{total_t}", end="")
    # 保存模型参数
    paddle.save(model.state_dict(), '../cifar10_test/cifar_10.pdparams')




if __name__ == '__main__':
    train(model)
