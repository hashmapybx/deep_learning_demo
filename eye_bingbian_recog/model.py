#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 17:27
# @Author  : Rocky
# @Site    : 
# @File    : model.py
# @Software: PyCharm


from paddle.vision.models import resnet18
import paddle.nn.functional as F
import paddle

loss_fn = F.cross_entropy

model = resnet18()
# print(paddle.summary(model,(1,3,224,224)))
