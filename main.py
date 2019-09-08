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
# image.reshape((5,6))

# todo 初始化图片
IMG = IMAGE(image)
# IMG.printimagesize()

# IMG.PERFORM()
IMG.REPERFORM()
# IMG.test()

# todo 图片加噪
# IMG.setgaussnoiseimage()
# showimage(IMG.getOriginimage(),IMG.getNoiseimage())


