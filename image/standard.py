# -*- coding: utf-8 -*-
'''
    @project:pag
    @file:standard.py
    @ide:PyCharm
    @time:2019-09-08 21:17
    @author:Sun
    @todo: 实现测试 PSNR(peak signal-to-noise ratio)
    @ref:
'''

import math


def calPSNR(sourceimage, targetimage):   #todo 问题：目标图像的范围不在0-255之间
    '''
    :param sourceimage:  源图像 都是灰度图，且是列表
    :param targetimage:  目标图像
    :return:
    '''
    length = len(sourceimage)
    width = len(sourceimage[0])

    # 计算MSE
    MSE = 0
    for i in range(length):
        for j in range(width):
            MSE += pow(sourceimage[i][j] - targetimage[i][j], 2)
    # print(MSE)
    MSE /= float(length * width)

    PSNR = 20 * math.log10(255 / math.sqrt(MSE))
    return PSNR


if __name__=='__main__':
    a=[[1,2,3,4]]
    b=[[2,3,4,5]]
    print(calPSNR(a,b))