#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/19 16:38
# @Author  : Rocky
# @Site    : 
# @File    : 礼帽黑帽.py
# @Software: PyCharm

# 礼帽：原始输入-开源算的结果
# 黑帽：闭运算- 原始输入

import cv2
import numpy as  np
img = cv2.imread(r"D:\pycharm_project\deep_learning_demo\opencv_test\test01.png")
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((3, 3), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('tophat', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()