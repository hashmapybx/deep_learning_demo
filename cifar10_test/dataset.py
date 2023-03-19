#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/17 09:06
# @Author  : Rocky
# @Site    : 
# @File    : dataset.py
# @Software: PyCharm

from paddle.vision.datasets import Cifar10
from paddle.vision.transforms import Transpose,Compose,Resize,CenterCrop,ToTensor,Normalize
import paddle
import matplotlib.pyplot as plt
import numpy as np


BATCH_SIZE=128


transform = Compose([
    Resize(32),
    # CenterCrop(32),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



train_dataset = Cifar10(mode='train', transform=transform,download=True)
test_dataset = Cifar10(mode='test', transform=transform,download=True)
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


trainloader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
testloader = paddle.io.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# # for batch_id,data in enumerate(train_loader()):
# #     print(data)
# #     break
# dataiter = iter(trainloader)
# images, labels = dataiter.next()


# def imgshow(img):
#     img = img/2. + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()












