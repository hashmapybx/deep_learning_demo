#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/19 16:44
# @Author  : Rocky
# @Site    : 
# @File    : sobel.py
# @Software: PyCharm

import cv2
img = cv2.imread(r"D:\pycharm_project\deep_learning_demo\opencv_test\aa.png")

def cv_show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



soebl_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
so_x = cv2.convertScaleAbs(soebl_x)
cv_show(so_x)

# 计算y轴
soebl_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
so_y = cv2.convertScaleAbs(soebl_y)
cv_show(so_y)

#  求和
so_xy = cv2.addWeighted(so_x,0.5,so_y,0.5,0)
cv_show(so_xy)



# 用sobel算子来获取图像中物体的轮廓线
luo = cv2.imread(r"D:\pycharm_project\deep_learning_demo\opencv_test\luo.png",0)
cv_show(luo)
luo_so_x = cv2.Sobel(luo,cv2.CV_64F,1,0,ksize=3)
luo_so_x = cv2.convertScaleAbs(luo_so_x)
luo_so_y = cv2.Sobel(luo,cv2.CV_64F,0,1,ksize=3)
luo_so_y = cv2.convertScaleAbs(luo_so_y)
luo_xy = cv2.addWeighted(luo_so_x,0.5,luo_so_y,0.5,0)
cv_show(luo_xy)

# 看一下scharr算子
scharr_x = cv2.Scharr(luo, cv2.CV_64F,1,0)
luo_scharr_x = cv2.convertScaleAbs(scharr_x)
scharr_y = cv2.Scharr(luo, cv2.CV_64F,0,1)
luo_scharr_y = cv2.convertScaleAbs(scharr_y)
luo_scharr_xy = cv2.addWeighted(luo_scharr_x,0.5,luo_scharr_y,0.5,0)
cv_show(luo_scharr_xy)