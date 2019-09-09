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
import numpy as np
import copy
import math
import time
import random

from cipher.cryption import Cryption
import init


def showimage(image, nosieimage):
    from matplotlib import pyplot as plt
    import pylab
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(image, cmap='gary')
    plt.title("Origin picture")
    plt.subplot(122)
    plt.imshow(nosieimage, cmap='gary')
    plt.title("Add Gaussian image")
    pylab.show()


class IMAGE(Cryption):
    def __init__(self, image):
        self.__image = copy.deepcopy(image)
        self.__grayimage = []
        self.__regrayimage=[]
        self.__length = len(image)
        self.__width = len(image[0])
        self.__size = self.__length * self.__width  # 像素个数

        # TODo 加密
        Cryption.__init__(self, init.M)
        self.__encryimage = []

        # todo JL
        self.__mat = []
        self.__S = init.S  # 每个像素的小矩阵的宽度
        self.__L = init.L
        self.__K = init.K  # K<=S^2
        # self.__R = init.R  # 生成的随机矩阵的元素值[0,R]
        self.__pNN = []  # 所有像素的像素块矩阵
        self.__P = []  # 随机矩阵
        self.__Y = None  # (length*width)*K
        self.__scope = self.__S // 2
        self.__lscope = self.__L // 2
        self.__relength = self.__length - self.__scope * 2
        self.__rewidth = self.__width - self.__scope * 2
        self.__resize = self.__relength * self.__rewidth
        self.__sigma = init.sigma  # 添加高斯噪音的标准差
        self.__JL = []  # (length*width)*K

        # todo 去噪
        self.__H = init.H  #
        self.__SCAL = init.SCAL
        self.__exp = []
        self.__Z = []
        self.__W = []
        self.__denoiseimage = None

        # todo 解密
        self.__decryimage = None

        # todo 评估
        self.__PSNR = 0

    # TODO 执行整个过程
    def PERFORM(self):
        import matplotlib.pyplot as plt

        stime = time.time()

        # todo 原图变为灰度图
        self.__setgrayimage()
        print('**********get grayimage!')
        plt.imshow(self.__grayimage, cmap='gray')
        plt.show()
        # self.__addgaussnoise()
        # print('**********grayimage add gauss noise!')
        # plt.imshow(self.__grayimage, cmap='gray')
        # plt.show()

        # todo 给灰度图的每个像素加密
        stime1 = time.time()
        self.__encryption()
        print('**********encry success!')
        etime1 = time.time()
        print('图像加密时间是：', etime1 - stime1, 's')

        # todo JL transform
        stime1 = time.time()
        self.__JLgetY()  # 得到JL变化的Y矩阵
        self.__JLaddgauss()  # 得到JL变换的最终矩阵
        self.__DNsetEXP()
        self.__DNsetZ()
        print('**********Z cal done!')
        self.__DNsetW()
        print('**********W cal done!')
        etime1 = time.time()
        print('JL变换时间是：', etime1 - stime1, 's')

        # TODO 对加密的图像进行去噪
        stime1 = time.time()
        self.__DNdenosing()
        print('**********denosing done!')
        etime1 = time.time()
        print('图像去噪时间是：', etime1 - stime1, 's')

        # TODO 对去噪后的图像进行解密
        stime1 = time.time()
        self.__decryption()
        print('**********decryption done!')
        etime1 = time.time()
        print('图像解密时间是：', etime1 - stime1, 's')

        etime = time.time()
        print('时间是：', etime - stime, 's')

        # print('解密后的图片：\n', self.__decryimage)
        plt.imshow(self.__decryimage, cmap='gray')
        plt.title('H=' + str(self.__H) + ' ,scal=' + str(self.__SCAL))
        plt.savefig('resimage/result: H=' + str(self.__H) + '-SCAL=' + str(self.__SCAL) + '.png')
        plt.show()

        # TODO 计算PSNR
        self.__calPSNR()

        print('done!')

    # TODO 测试在不加入加解密的情况下的去噪结果与加入加解密的结果对比
    def REPERFORM(self):
        import matplotlib.pyplot as plt

        stime = time.time()

        # todo 原图变为灰度图
        self.__setgrayimage()
        print('**********get grayimage!')
        self.__regrayimage=self.__grayimage

        plt.imshow(self.__grayimage, cmap='gray')
        plt.show()

        self.__addgaussnoise()
        print('**********grayimage add gauss noise!')
        plt.imshow(self.__grayimage, cmap='gray')
        plt.show()
        print("加噪后的灰度图",self.__grayimage)
        # return

        # todo 给灰度图的每个像素加密
        stime1 = time.time()
        self.__encryption()
        print('**********encry success!')
        etime1 = time.time()
        print('图像加密时间是：', etime1 - stime1, 's')

        # todo JL transform
        stime1 = time.time()
        self.__JLgetY()  # 得到JL变化的Y矩阵
        self.__JLaddgauss()  # 得到JL变换的最终矩阵
        self.__DNsetEXP()
        self.__DNsetZ()
        print('**********Z cal done!')
        self.__DNsetW()
        print('**********W cal done!')
        etime1 = time.time()
        print('JL变换时间是：', etime1 - stime1, 's')

        # TODO 对未加密的图像进行去噪
        stime1 = time.time()
        self.__ReDNdenosing()
        print('**********denosing done!')
        etime1 = time.time()
        plt.imshow(self.__denoiseimage, cmap='gray')
        plt.title('none encryption: H=' + str(self.__H) + ' ,scal=' + str(self.__SCAL))
        plt.savefig('resimage/none encryption result: H=' + str(self.__H) + '-SCAL=' + str(self.__SCAL) + '.png')
        plt.show()
        print('未加密图像去噪时间是：', etime1 - stime1, 's')
        self.__calPSNR2()
        print("为加密的图像",self.__denoiseimage)

        # TODO 对加密的图像进行去噪
        stime1 = time.time()
        self.__DNdenosing()
        print('**********denosing done!')
        etime1 = time.time()
        print('图像去噪时间是：', etime1 - stime1, 's')

        # TODO 对去噪后的图像进行解密
        stime1 = time.time()
        self.__decryption()
        print('**********decryption done!')
        etime1 = time.time()
        print('图像解密时间是：', etime1 - stime1, 's')

        etime = time.time()
        print('时间是：', etime - stime, 's')

        print('解密后的图片：\n', self.__decryimage)
        plt.imshow(self.__decryimage, cmap='gray')
        plt.title('encryption H=' + str(self.__H) + ' ,scal=' + str(self.__SCAL))
        plt.savefig('resimage/encryption result: H=' + str(self.__H) + '-SCAL=' + str(self.__SCAL) + '.png')
        plt.show()

        # TODO 计算PSNR
        self.__calPSNR()

        print('done!')

    # todo 计算去噪图片和原始图片的PSNR
    def __calPSNR(self):
        sourceimage = [i[self.__scope:-self.__scope] for i in self.__regrayimage][self.__scope:-self.__scope]
        from image.standard import calPSNR
        self.__PSNR = calPSNR(sourceimage, self.__decryimage)
        print('图像的PSNR值为: ', self.__PSNR)

    # todo 计算未加密去噪图片和原始图片的PSNR
    def __calPSNR2(self):
        sourceimage = [i[self.__scope:-self.__scope] for i in self.__regrayimage][self.__scope:-self.__scope]
        from image.standard import calPSNR
        self.__PSNR = calPSNR(sourceimage, self.__denoiseimage)
        print('图像的PSNR值为: ', self.__PSNR)

    # TODO 对原图像进行加密
    def __encryption(self):
        self.reinit()
        self.__encryimage = [None] * self.__width
        self.__encryimage = [self.__encryimage[:] for i in range(self.__length)]
        for i in range(self.__length):
            # s = time.time()
            for j in range(self.__width):
                self.__encryimage[i][j] = self.REencryption(self.__grayimage[i][j])  # 对每个
                # print(self.__encryimage[i][j])
            # t = time.time()
            # print(t - s)

    def __decryption2(self):
        self.__decryimage = [None] * self.__width
        self.__decryimage = [self.__decryimage[:] for i in range(self.__length)]
        for i in range(self.__length):
            for j in range(self.__width):
                self.__decryimage[i][j] = self.REdecryption(self.__encryimage[i][j])

    def __decryption(self):
        self.__decryimage = [None] * self.__rewidth
        self.__decryimage = [self.__decryimage[:] for i in range(self.__relength)]
        for i in range(self.__relength):
            for j in range(self.__rewidth):
                self.__decryimage[i][j] = self.REdecryption(self.__denoiseimage[i][j]) // self.__SCAL
                # print(self.__decryimage[i][j])

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

    # todo 为灰度图添加高斯噪声
    def __addgaussnoise(self):
        for i in range(self.__length):
            for j in range(self.__width):
                # gauss=random.gauss(mu=0, sigma=self.__sigma)
                # print(gauss)
                self.__grayimage[i][j] += int(random.gauss(mu=0, sigma=self.__sigma))

    # TODO 得到JL变换中的p矩阵，P矩阵本身是一个高斯矩阵
    def __JLgetP(self):  # 生成变换矩阵P，是一个随机矩阵
        self.__P = np.random.normal(0, 1 / self.__K, [self.__S ** 2, self.__K])

    def __JLgetpixelNN(self):
        tag = 0
        for ti in range(self.__scope, self.__length - self.__scope):
            for tj in range(self.__scope, self.__width - self.__scope):
                # self.__mat.append([])
                self.__pNN.append([])
                ind = ti * self.__width + tj
                beg = ind - self.__width * self.__scope - self.__scope  # 开始位置的像素
                # print('\n', beg)
                for i in range(self.__S):
                    for j in range(self.__S):
                        indd = beg + i * self.__width + j
                        # self.__mat[tag].append(indd)
                        self.__pNN[tag].append(self.__grayimage[indd // self.__width][indd % self.__width])
                tag += 1
        # print(tag)

    # todo 得到JL变换中的Y矩阵
    def __JLgetY(self):
        self.__JLgetP()
        self.__JLgetpixelNN()
        from cipher.chRT import matmulti
        self.__Y = matmulti(self.__pNN, self.__P, len(self.__pNN), self.__S ** 2, self.__K)
        # self.__Y = np.dot(self.__pNN, self.__P)

    # todo JL变换中对Y矩阵添加高斯噪声
    def __JLaddgauss(self):
        self.__JL = [0] * self.__K
        self.__JL = [self.__JL[:] for i in range(self.__resize)]
        for i in range(self.__resize):
            for j in range(self.__K):
                self.__JL[i][j] = self.__Y[i][j]
                # self.__JL[i][j] = self.__Y[i][j] + random.gauss(mu=0, sigma=self.__sigma)

    def __DNsetEXP(self, step=init.step):  # todo 注意：以空间换时间的方法，图片太大会爆内存
        self.__exp = [0] * self.__resize
        self.__exp = [self.__exp[:] for i in range(self.__resize)]
        TMPC = 2 * self.__K * pow(self.__sigma, 2)

        def getexp(i, TMPC):
            for j in range(i + 1, self.__resize):
                tmpdis = [abs(x - y) for x, y in zip(self.__JL[i], self.__JL[j])]  # 欧氏距离
                tmpdis = list(map(lambda x: pow(x, 2), tmpdis))
                tmpsum = sum(tmpdis)  # k维向量的每一个值加和
                # tem = -(tmpsum - TMPC) / pow(self.__H, 2)
                tem = -tmpsum / pow(self.__H, 2)
                tem = math.exp(tem)
                # tem = int(self.__SCAL * tem)
                # if tem == 0:
                #     print(i, j)
                # print(tem)
                self.__exp[i][j] = tem
                self.__exp[j][i] = tem

        from threading import Thread
        for i in range(0, self.__resize, step):
            # s = time.time()
            func = []
            for j in range(step):
                if j < self.__resize:
                    func.append(Thread(target=getexp, args=(i + j, TMPC)))
            for th in func:
                th.start()
            for th in func:
                th.join()
        # t = time.time()
        # print('iter %d time is %f s' % (i, (t - s)))

        """for i in range(self.__resize):
            # print(i)
            for j in range(i + 1, self.__resize):
                tmpdis = [abs(x - y) for x, y in zip(self.__JL[i], self.__JL[j])]  # 欧氏距离
                tmpdis = list(map(lambda x: pow(x, 2), tmpdis))
                tmpsum = sum(tmpdis)  # k维向量的每一个值加和
                # tem = -(tmpsum - TMPC) / pow(self.__H, 2)
                tem = -tmpsum / pow(self.__H, 2)
                tem = math.exp(tem)
                tem = int(self.__SCAL * tem)
                self.__exp[i][j] = tem
                self.__exp[j][i] = tem"""

        # for i in range(self.__resize):
        #     print(i, self.__exp[i])

    def __DNsetZ(self):
        self.__mat = []
        for ti in range(self.__relength):
            for tj in range(self.__rewidth):
                self.__mat.append([])
                ind = ti * self.__rewidth + tj
                jbegin = max(0, tj - self.__lscope)
                jend = min(tj + self.__lscope + 1, self.__rewidth)
                ibegin = max(0, ti - self.__lscope)
                iend = min(ti + self.__lscope + 1, self.__relength)
                for i in range(ibegin, iend):
                    for j in range(jbegin, jend):
                        self.__mat[ind].append(i * self.__rewidth + j)

        for i in range(self.__resize):
            tsum = 0
            for j in self.__mat[i]:
                # if j >= 306:
                #     print(j)
                # if j == -1:
                #     continue
                tsum += self.__exp[i][j]
            # print(i, tsum)
            self.__Z.append(tsum)

        # print('mat:')
        # for i in range(len(self.__mat)):
        #     print(i, self.__mat[i])
        #
        # print(self.__Z)

    # TODO 得到JL变换的去噪参数
    def __DNsetW(self):
        self.__W = [0] * self.__resize
        self.__W = [self.__W[:] for i in range(self.__resize)]
        for i in range(self.__resize):
            for j in range(self.__resize):
                # if self.__Z[i] == 0:
                #     self.__W[i][j] = 0
                #     continue
                self.__W[i][j] = self.__exp[i][j] / self.__Z[i]  # i=j的时候怎么办，现在是0
                self.__W[i][j] = int(self.__SCAL * self.__W[i][j])
        # for i in range(self.__W.__len__()):
        #     print(self.__W[i])

    # TODO 对加密后的图像进行去噪
    def __DNdenosing(self):
        self.__denoiseimage = [0] * self.__rewidth
        self.__denoiseimage = [self.__denoiseimage[:] for i in range(self.__relength)]
        encryimage = [i[self.__scope:-self.__scope] for i in self.__encryimage][self.__scope:-self.__scope]
        # print(encryimage.__len__())
        # print(encryimage[0].__len__())
        for i in range(self.__relength):
            for j in range(self.__rewidth):
                sumz = [[0, 0, 0, 0][:] for i in range(4)]
                ind = i * self.__rewidth + j
                for l in self.__mat[ind]:
                    for ii in range(4):
                        for jj in range(4):
                            sumz[ii][jj] += self.__W[ind][l] * encryimage[l // self.__rewidth][l % self.__rewidth][ii][
                                jj]
                # print(sumz)
                self.__denoiseimage[i][j] = sumz

    # TODO 对原始图像进行去噪，用于与加密图像去噪后作对比
    def __ReDNdenosing(self):
        self.__denoiseimage = [0] * self.__rewidth
        self.__denoiseimage = [self.__denoiseimage[:] for i in range(self.__relength)]
        encryimage = [i[self.__scope:-self.__scope] for i in self.__grayimage][self.__scope:-self.__scope]
        # print(encryimage.__len__())
        # print(encryimage[0].__len__())
        for i in range(self.__relength):
            for j in range(self.__rewidth):
                sumz = 0
                ind = i * self.__rewidth + j
                for l in self.__mat[ind]:
                    sumz += self.__W[ind][l] * encryimage[l // self.__rewidth][l % self.__rewidth]
                # print(sumz)
                self.__denoiseimage[i][j] = sumz // self.__SCAL

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
        return self.__denoiseimage

    def getGrayimage(self):
        return self.__grayimage

    def printimagesize(self):
        print('height:', self.__length)
        print('width:', self.__width)
