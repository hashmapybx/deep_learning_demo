#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 17:28
# @Author  : Rocky
# @Site    : 
# @File    : runn.py
# @Software: PyCharm

from eye_bingbian_recog.dataset import data_loader, valid_data_loader
import paddle
import numpy as np
import paddle.nn.functional as F
from eye_bingbian_recog.model import model, loss_fn
import os


class Runner(object):
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # 记录全局最优指标
        self.best_acc = 0

    # 定义训练过程
    def train(self, train_datadir, **kwargs):
        print("\t start training.....\t")
        self.model.train()
        num_epochs = kwargs.get("epoch", 0)
        # csv_file = kwargs.get('csv_file',0)
        save_path = kwargs.get('save_path', '../eye_bingbian_recog/model/')

        # 定义数据读取器，训练数据读取器
        train_loader = data_loader(train_datadir, batch_size=10, mode='train')

        for epoch in range(num_epochs):
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                img = paddle.to_tensor(x_data)
                label = paddle.to_tensor(y_data)

                # 前向计算
                pred = self.model(img)
                avg_loss = self.loss_fn(pred, label)

                if batch_id % 20 == 0:
                    print("epoch:{} ,batch_id:{}, losss is: {}".format(epoch, batch_id, float(avg_loss.numpy())))
                # 反馈后向计算
                avg_loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
        self.save_model(save_path)

    # 模型评估阶段，使用'paddle.no_grad()'控制不计算和存储梯度
    @paddle.no_grad()
    def evaluate_pm(self, val_datadir, csv_file):
        self.model.eval()
        accuracies = []
        losses = []
        # 验证数据读取器
        valid_loader = valid_data_loader(val_datadir, csv_file)

        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            # 运行模型前向计算，得到预测值
            logits = self.model(img)
            # 多分类，使用softmax计算预测概率
            pred = F.softmax(logits)
            loss = self.loss_fn(pred, label)
            acc = paddle.metric.accuracy(pred, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
        return np.mean(accuracies)

    # 模型评估阶段，使用'paddle.no_grad()'控制不计算和存储梯度
    @paddle.no_grad()
    def predict_pm(self, x, **kwargs):
        # 将模型设置为评估模式
        self.model.eval()
        # 运行模型前向计算，得到预测值
        logits = self.model(x)
        return logits

    def save_model(self, save_path):
        paddle.save(self.model.state_dict(), save_path + 'palm.pdparams')
        paddle.save(self.optimizer.state_dict(), save_path + 'palm.pdopt')

    def load_model(self, model_path):
        model_state_dict = paddle.load(model_path)
        self.model.set_state_dict(model_state_dict)


if __name__ == '__main__':
    # 开启0号GPU训练
    use_gpu = True
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

    # 定义优化器
    # opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters(), weight_decay=0.001)
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    runner = Runner(model, opt, loss_fn)

    # 数据集路径
    DATADIR = r"D:\other\eye_chanllege_competition\training\PALM-Training400\PALM-Training400"
    DATADIR2 = r'/home/aistudio/work/palm/PALM-Validation400'
    CSVFILE = r'D:\other\eye_chanllege_competition\labels.csv'
    # 设置迭代轮数
    EPOCH_NUM = 5
    # 模型保存路径
    PATH = '../eye_bingbian_recog/model/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    # 启动训练过程
    runner.train(DATADIR, DATADIR2,
                 num_epochs=EPOCH_NUM, csv_file=CSVFILE, save_path=PATH)
