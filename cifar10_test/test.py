#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/18 18:28
# @Author  : Rocky
# @Site    : 
# @File    : test.py
# @Software: PyCharm

import paddle
from cifar10_test.dataset import testloader, BATCH_SIZE
import numpy as np
import paddle.nn.functional as F
from cifar10_test.model import model
import matplotlib.pyplot as  plt

def evaluation(model):
    print('start evaluation .......')
    # 定义预测过程
    params_file_path = '../cifar10_test/cifar_10.pdparams'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)
    model.eval()
    vote_correct = 0
    test_sum  = 0
    for img, label in testloader():
        predicts= model(img)
        prediction = paddle.argmax(predicts,1)
        pre_num = prediction.cpu().numpy()
        vote_correct += (pre_num == label.cpu().numpy()).sum()
        test_sum += BATCH_SIZE
        print(
            f"\rTest ACC： {(vote_correct / test_sum):.5}",
            end="")


def test_one(model):
    # 获取测试集的一张图片
    x,label = next(testloader())
    print(x[0].shape)
    print(label[0].numpy()[0])
    params_file_path = '../cifar10_test/cifar_10.pdparams'
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)
    model.eval()
    pred = model(x)
    # 获取类别最大的概率
    p_num = paddle.argmax(pred,1).cpu().numpy()
    print("p_num", p_num[0])
    print("label", label[0].cpu().numpy())
    print("在第一个batch里面的准确率是: ", (p_num == label.cpu().numpy).sum()/label.shape[0])


    plt.figure(figsize=(2,2))
    plt.imshow(x[0].numpy().astype('float32').transpose(1,2,0))
    plt.show()


if __name__ == '__main__':
    test_one(model)
