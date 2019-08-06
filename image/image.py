# -*- coding: utf-8 -*-
'''
    @project:pag
    @file:image.py
    @ide:PyCharm
    @time:2019-08-03 22:16
    @author:Sun
    @todo:
    @ref:
'''
from image.noise import *
import numpy as np
import copy


class IMAGE:

    def __init__(self, image, S,K):
        self.image = copy.deepcopy(image)
        self.noiseimage = None
        self.grayimage = []
        self.__length = len(image)
        self.__width = len(image[0])
        self.__pnum = self.__length * self.__width  # 像素个数

        # todo JL
        self.__S = S
        self.__K = K  # K<=S^2
        self.__P = []
        self.Y = None

    # 得到椒盐加噪的图片
    def getsaltnoiseimage(self, proportion):
        """
        :param proportion: 加噪点占总像素的比例，默认0.05
        :return:
        """
        self.noiseimage = salt_and_pepper_noise(copy.deepcopy(self.image), proportion=proportion)

    # 得到高斯加噪的图片
    def getgaussnoiseimage(self, sigma):
        """
        :param sigma: sigma值,默认100
        :return:
        """
        self.noiseimage = gauss_noise(copy.deepcopy(self.image), sigma=sigma)

    # 得到加噪后图片的灰度图
    def getgrayimage(self):
        for i in range(self.__length):
            self.grayimage.append([])
            for j in range(self.__width):
                r = self.noiseimage[i][j][0]
                g = self.noiseimage[i][j][1]
                b = self.noiseimage[i][j][2]
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                self.grayimage[i].append(gray)

    def JLgetP(self, R):  # 生成变换矩阵P，是一个随机矩阵
        """
        :param R: p矩阵是由0-R的数组成的随机矩阵
        :return:
        """
        for i in range(self.__S ** 2):
            self.__P.append([])
            for j in range(self.__K):
                self.__P[i].append(random.random() * R)

    def __JLgetpixelNN(self):
        def JLgetNN(ind):
            res = []
            scope = self.__S // 2
            beg = ind - self.__width * scope - scope  # 开始位置的像素
            for i in range(self.__S):
                for j in range(self.__S):
                    indd = beg + i * self.__width + j
                    if indd < 0 or indd >= self.__pnum:
                        res.append(0)
                    else:
                        res.append(self.grayimage[indd // self.__width][indd % self.__width])
            return res

        pNN = []
        for i in range(self.__pnum):
            pNN.append(JLgetNN(i))
        return pNN

    def JLgetY(self):
        self.Y = np.dot(self.__JLgetpixelNN(), self.__P).tolist()


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
