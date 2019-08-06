# -*- coding: utf-8 -*-
'''
    @project:pag
    @file:main.py
    @ide:PyCharm
    @time:2019-08-03 22:08
    @author:Sun
    @todo:
    @ref:
'''

from skimage import io
import init
from image.image import *
import matplotlib.pylab as plt
import numpy as np

# todo 读取图片
image = io.imread(init.imagepath)  # length*width*(r,g,b)
print(image)
# image=np.array([[17, 44, 169, 126], [91, 121, 84, 85], [85, 71, 119, 25], [0, 85, 201, 44]])

# todo 初始化图片
IMG = IMAGE(image, init.S, init.K)

# todo 图片加噪
IMG.getsaltnoiseimage(proportion=init.proportion)
# showimage(IMG.image,IMG.noiseimage)


# todo 获取加噪图像灰度
IMG.getgrayimage()  # length*width*gray
# plt.imshow(IMG.grayimage)
# plt.show()

# todo JL变换
IMG.JLgetP(0.5)
IMG.JLgetY()
print(IMG.Y)
