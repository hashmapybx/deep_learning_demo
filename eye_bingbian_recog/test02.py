#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 19:49
# @Author  : Rocky
# @Site    : 
# @File    : test02.py
# @Software: PyCharm

from eye_bingbian_recog.model import model, loss_fn
from eye_bingbian_recog.dataset import valid_data_loader,transform_img
import paddle
import paddle.nn.functional as F
import numpy as np
import os,cv2


def test():
    # 使用验证集来验证模型在训练集上面的表现
    params_file_path = '../eye_bingbian_recog/eye_01.pdparams'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)
    model.eval()

    accuracies = []
    losses = []
    # 验证数据读取器
    val_datadir = r"D:\other\eye_chanllege_competition\validation\PALM-Validation400"
    csv_file = r'D:\other\eye_chanllege_competition\labels.csv'
    valid_loader = valid_data_loader(val_datadir, csv_file)

    for batch_id, data in enumerate(valid_loader()):
        x_data, y_data = data
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)
        # 运行模型前向计算，得到预测值
        logits = model(img)
        # 多分类，使用softmax计算预测概率
        pred = F.softmax(logits)
        loss = loss_fn(pred, label)
        acc = paddle.metric.accuracy(pred, label)
        accuracies.append(acc.numpy())
        losses.append(loss.numpy())
    print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))


def predict_one_picture():
    # 加载测试集上的一张图片
    val_datadir = r"D:\other\eye_chanllege_competition\validation\PALM-Validation400"
    csv_file = r'D:\other\eye_chanllege_competition\labels.csv'
    filelists = open(csv_file).readlines()
    line = filelists[1].strip().split(',')
    print(line)
    name,label = line[1],int(line[2])

    # 读取图片
    img = cv2.imread(os.path.join(val_datadir,name))
    # 预测图片预处理
    img = transform_img(img)
    unsqueeze_img = paddle.unsqueeze(paddle.to_tensor(img), axis=0) # 添加一个维度

    params_file_path = '../eye_bingbian_recog/eye_01.pdparams'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)
    model.eval()

    pre = model(unsqueeze_img)
    result  = F.softmax(pre)
    pre_class = paddle.argmax(result).numpy()

    #输出真实类别和预测类别
    print("the true label is {}, the predict label is {}".format(label, pre_class))


if __name__ == '__main__':
    predict_one_picture()
