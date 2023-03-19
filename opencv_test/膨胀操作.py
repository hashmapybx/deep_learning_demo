#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/19 16:00
# @Author  : Rocky
# @Site    : 
# @File    : 膨胀操作.py
# @Software: PyCharm

import cv2
import numpy as np

# 膨胀操作

img = cv2.imread(r"D:\pycharm_project\deep_learning_demo\opencv_test\test01.png")
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((3, 3), np.uint8)
erode = cv2.erode(img, kernel, iterations=1)  # 腐蚀
# cv2.imshow('erode', erode)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 膨胀
# dige = cv2.dilate(erode, kernel, iterations=1)
# cv2.imshow('dilate', dige)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# read 图片
img2 = cv2.imread(r"D:\pycharm_project\deep_learning_demo\opencv_test\aa.png")

# kernel_01 = np.ones((30,30), np.uint8)
# erode_1 = cv2.erode(img2,kernel_01,iterations =1)
# erode_2 = cv2.erode(img2,kernel_01,iterations =2)
# erode_3 = cv2.erode(img2,kernel_01,iterations =3)
# res = np.hstack((erode_1,erode_2,erode_3))
# cv2.imshow('res',res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 开运算 先腐蚀 再膨胀
mor = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('mor', mor)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 闭运算 先膨胀 在腐蚀 这种就不能去除图像中毛刺边缘
close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('close', close)
cv2.waitKey(0)
cv2.destroyAllWindows()




