#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 16:58
# @Author  : Rocky
# @Site    : 
# @File    : imgh_show.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import os
from PIL import Image
import  numpy as np
img_dir_path= r"D:\other\eye_chanllege_competition\training\PALM-Training400\PALM-Training400"

file1 = "H0001.jpg"
file2 = "H0002.jpg"

# 读取图片
img1 = Image.open(os.path.join(img_dir_path, file1))
img1 = np.array(img1)
img2 = Image.open(os.path.join(img_dir_path, file2))
img2 = np.array(img2)

# 画出读取的图片
plt.figure(figsize=(16, 8))
f = plt.subplot(121)
f.set_title('Normal', fontsize=20)
plt.imshow(img1)
f = plt.subplot(122)
f.set_title('PM', fontsize=20)
plt.imshow(img2)
plt.show()

print(img1.shape,img2.shape)

