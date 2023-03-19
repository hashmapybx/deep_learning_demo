#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/19 16:31
# @Author  : Rocky
# @Site    : 
# @File    : 梯度运算.py
# @Software: PyCharm
import cv2
import numpy as np

img2 = cv2.imread(r"D:\pycharm_project\deep_learning_demo\opencv_test\aa.png")
kernel = np.ones((7,7), np.uint8)
dliate = cv2.dilate(img2,kernel,iterations=5)
erode = cv2.erode(img2,kernel,iterations=5)
res = np.hstack((dliate,erode))
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 计算梯度 = 膨胀-腐蚀
gradient = cv2.morphologyEx(img2, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()