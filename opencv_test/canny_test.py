#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/19 18:13
# @Author  : Rocky
# @Site    : 
# @File    : canny_test.py
# @Software: PyCharm


"""opencv里面canny检测"""
import cv2
import numpy as np
from opencv_test.sobel import cv_show

luo = cv2.imread(r"D:\pycharm_project\deep_learning_demo\opencv_test\luo.png",0)
c1 = cv2.Canny(luo,80,150) # 80 150 代表的是双阈值检测的最大值和最小值
c2 = cv2.Canny(luo,50,10)
res = np.hstack((c1,c2))
cv_show(res)



