# -*- coding: utf-8 -*-
'''
    @project:pag
    @file:init.py
    @ide:PyCharm
    @time:2019-07-28 20:16
    @author:Sun
'''

# todo image
imagepath = 'img/000.jpg'
# imagepath = 'img/003.png'

# todo JL
L = 7 # JL变换中滤波窗口的大小   21
S = 3  # Jl变换中像素的小矩阵的宽度   5
K = 4  # JL变换降维后的维度   18
# R = 0.5  # JL变换中随机矩阵P的元素的范围，其中每个元素的大小在0-R之间
sigma = 60  # JL变换中添加高斯噪声的标准差，范围是N(0,sigma^2)
step = 4  # 计算eij的线程数

# todo encryption
M = 10  # 加密算法的M值

# todo 去噪
H = 40 # 应该是k的10倍 180
SCAL = 1000
