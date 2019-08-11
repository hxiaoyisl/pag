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
import init
import math
import time
from cipher.cryption import Cryption


class IMAGE(Cryption):

    def __init__(self, image):
        self.__image = copy.deepcopy(image)
        self.__noiseimage = None
        self.__grayimage = []
        self.__length = len(image)
        self.__width = len(image[0])
        self.__size = self.__length * self.__width  # 像素个数

        # TODo 加密
        Cryption.__init__(self, init.M)
        self.__encryimage = []

        # todo JL
        self.__mat = []
        self.__S = init.S  # 每个像素的小矩阵的宽度
        self.__K = init.K  # K<=S^2
        self.__R = init.R  # 生成的随机矩阵的元素值[0,R]
        self.__pNN = []  # 所有像素的像素块矩阵
        self.__P = []  # 随机矩阵
        self.__Y = None  # (length*width)*K
        self.__sigma = init.sigma  # 添加高斯噪音的标准差
        self.__JL = []  # (length*width)*K

        # todo 去噪
        self.__H = init.H
        self.__exp = []
        self.__Z = []
        self.__W = []

    def Init(self):
        import matplotlib.pyplot as plt
        self.__setgrayimage()  # 原图变为灰度图
        print('**********get grayimage!')
        # plt.imshow(self.__grayimage)
        # plt.show()

        self.__encryption()  # 给灰度图的每个像素加密
        print('**********encry success!')

        self.__JLgetY()  # 得到JL变化的Y矩阵
        # print('**********get JL Y!')
        self.__JLaddgauss()  # 得到JL变换的最终矩阵
        print('**********JL transform done!')
        return
        self.__DNsetZ()
        print('**********Z cal done!')
        self.__DNsetW()
        print('**********W cal done!')
        print('done!')

    # TODO 对原图像进行加密
    def __encryption(self):
        self.reinit()
        self.__encryimage = [None] * self.__width
        self.__encryimage = [self.__encryimage[:] for i in range(self.__length)]
        for i in range(self.__length):
            # s = time.time()
            for j in range(self.__width):
                self.__encryimage[i][j] = self.REencryption(self.__grayimage[i][j])  # 对每个
            # t = time.time()
            # print(t - s)

    # todo 原图变为灰度图
    def __setgrayimage(self):
        for i in range(self.__length):
            self.__grayimage.append([])
            for j in range(self.__width):
                r = self.__image[i][j][0]
                g = self.__image[i][j][1]
                b = self.__image[i][j][2]
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                self.__grayimage[i].append(gray)

    def __JLgetP(self):  # 生成变换矩阵P，是一个随机矩阵
        """
        :param R: p矩阵是由0-R的数组成的随机矩阵
        :return:
        """
        for i in range(self.__S ** 2):
            self.__P.append([])
            for j in range(self.__K):
                self.__P[i].append(random.random() * self.__R)

    def __JLgetpixelNN(self):
        scope = self.__S // 2
        for ti in range(self.__length):
            for tj in range(self.__width):
                self.__mat.append([])
                self.__pNN.append([])
                ind = ti * self.__width + tj
                beg = ind - self.__width * scope - scope  # 开始位置的像素
                # print('\n', beg)
                for i in range(self.__S):
                    for j in range(self.__S):
                        indd = beg + i * self.__width + j
                        # print(indd, end=' ')
                        if indd < 0 or indd >= self.__size:
                            self.__mat[ind].append(-1)
                            self.__pNN[ind].append(0)
                        elif indd // self.__width != ti + i - 1:
                            self.__mat[ind].append(-1)
                            self.__pNN[ind].append(0)
                        else:
                            self.__mat[ind].append(indd)
                            self.__pNN[ind].append(self.__grayimage[indd // self.__width][indd % self.__width])

    # todo 得到JL变换中的Y矩阵
    def __JLgetY(self):
        self.__JLgetP()
        self.__JLgetpixelNN()
        self.__Y = np.dot(self.__pNN, self.__P)
        # self.__Y = np.dot(self.__pNN, self.__P).tolist()
        # print('JL变换的Y矩阵:\n',self.__Y)

    # todo JL变换中对Y矩阵添加高斯噪声
    def __JLaddgauss(self):
        for i in range(self.__size):
            self.__JL.append([])
            for j in range(self.__K):
                tmp = self.__Y[i][j] + random.gauss(mu=0, sigma=self.__sigma)
                if tmp > 225:
                    tmp = 225
                elif tmp < 0:
                    tmp = 0
                self.__JL[i].append(tmp)
        np.array(self.__JL)
        # print(self.__JL)

    def __DNsetZ(self, step=init.step):
        self.__exp = [0] * self.__size
        self.__exp = [self.__exp[:] for i in range(self.__size)]
        TMPC = 2 * self.__K * pow(self.__sigma, 2)

        def getexp(i, TMPC):
            for j in range(i + 1, self.__size):
                # print(j, end=' ')
                tmpdis = np.abs(np.array(self.__JL[i]) - np.array(self.__JL[j]))  # 欧氏距离
                # print(tmpdis)
                tmpdis = pow(tmpdis, 2)  # 距离平方
                tmpsum = sum(tmpdis)  # k维向量的每一个值加和
                tem = -(tmpsum - TMPC) / pow(self.__H, 2)
                tem = math.exp(tem)
                # print(tem)
                self.__exp[i][j] = tem
                self.__exp[j][i] = tem
            # print(i, end=' ')

        from threading import Thread
        for i in range(0, self.__size, step):
            # print('\niterator: ', i)
            func = []
            for j in range(step):
                if j < self.__size:
                    func.append(Thread(target=getexp, args=(i + j, TMPC)))
            for th in func:
                th.start()
            for th in func:
                th.join()
        # print(self.__exp)
        """for i in range(self.__size):
            print(i)
            for j in range(i + 1, self.__size):
                # print(j, end=' ')
                tmpdis = np.abs(np.array(self.__JL[i]) - np.array(self.__JL[j]))  # 欧氏距离
                # print(tmpdis)
                tmpdis = pow(tmpdis, 2)  # 距离平方
                tmpsum = 0
                for el in tmpdis:
                    tmpsum += el
                tem = -(tmpsum - TMPC) / pow(self.__H, 2)
                tem = math.exp(tem)
                self.__exp[i][j] = tem
                self.__exp[j][i] = tem"""

        for i in range(self.__size):
            tsum = 0
            for j in self.__mat[i]:
                if j == -1:
                    continue
                tsum += self.__exp[i][j]
            self.__Z.append(tsum)

        # print(self.__Z)

    def __DNsetW(self):
        self.__W = [0] * self.__size
        self.__W = [self.__W[:] for i in range(self.__size)]
        for i in range(self.__size):
            for j in range(self.__size):
                self.__W[i][j] = self.__exp[i][j] / self.__Z[i]  # i=j的时候怎么办，现在是0

    def test(self):
        self.__grayimage = [[i * j for j in range(1, 6)] for i in range(1, 6)]
        self.__size = 25
        self.__length = 5
        self.__width = 5
        self.__JLgetY()
        print(self.__grayimage)
        print('mat:')
        for i in range(self.__size):
            print(i, self.__mat[i])
        print('pNN:', self.__pNN)
        print('Y:', self.__Y)
        self.__JLaddgauss()
        print(self.__JL)

    def getOriginimage(self):
        return self.__image

    def getNoiseimage(self):
        return self.__noiseimage

    def getGrayimage(self):
        return self.__grayimage

    def printimagesize(self):
        print('height:', self.__length)
        print('width:', self.__width)
