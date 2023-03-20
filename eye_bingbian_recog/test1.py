#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 19:38
# @Author  : Rocky
# @Site    : 
# @File    : test1.py
# @Software: PyCharm

from eye_bingbian_recog.model import model,loss_fn
from eye_bingbian_recog.dataset import data_loader
import paddle

# 开启0号GPU训练
use_gpu = True
paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

model.train()
opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

train_datadir = r"D:\other\eye_chanllege_competition\training\PALM-Training400\PALM-Training400"

train_loader = data_loader(train_datadir, batch_size=10, mode='train')
print(train_loader)
num_epochs = 5
for epoch in range(num_epochs):
    for batch_id, data in enumerate(train_loader()):
        x_data, y_data = data
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)

        # 前向计算
        pred = model(img)
        avg_loss = loss_fn(pred, label)

        if batch_id % 20 == 0:
            print("epoch:{} ,batch_id:{}, losss is: {}".format(epoch, batch_id, float(avg_loss.numpy())))
        # 反馈后向计算
        avg_loss.backward()
        opt.step()
        opt.clear_grad()

paddle.save(model.state_dict(), '../eye_bingbian_recog/eye_01.pdparams')


