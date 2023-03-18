#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/17 08:58
# @Author  : Rocky
# @Site    : 
# @File    : model.py
# @Software: PyCharm


# 定义resnet18
from paddle.vision.models import resnet18
import paddle

model = resnet18()
# paddle.summary(model,(1,3,32,32))




