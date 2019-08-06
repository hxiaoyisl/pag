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

def showimage(image, nosieimage):
    from matplotlib import pyplot as plt
    import pylab
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(image)
    plt.title("Origin picture")
    plt.subplot(122)
    plt.imshow(nosieimage)
    plt.title("Add Gaussian image")
    pylab.show()

# todo 读取图片
image = io.imread(init.imagepath)  # length*width*(r,g,b)

# todo 初始化图片
IMG = IMAGE(image)
IMG.Init()
# IMG.test()

# todo 图片加噪
# IMG.setgaussnoiseimage()
# showimage(IMG.getOriginimage(),IMG.getNoiseimage())


