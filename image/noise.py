# -*- coding: utf-8 -*-
'''
    @project:pag
    @file:image.py
    @ide:PyCharm
    @time:2019-08-03 10:39
    @author:Sun
    @todo:高斯噪音和椒盐噪音
    @ref:https://www.csdn.net/gather_27/MtjaUgysNzkzLWJsb2cO0O0O.html
'''
from skimage import io
import numpy as np
import random


# 添加椒盐噪声
def salt_and_pepper_noise(img, proportion=0.05):
    noise_img = img
    height, width = noise_img.shape[0], noise_img.shape[1]
    num = int(height * width * proportion)  # 多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img

# 添加高斯噪声
def gauss_noise(image,sigma=100):
    img = image.astype(np.int16)#此步是为了避免像素点小于0，大于255的情况
    mu =0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i,j,k] = img[i,j,k] +random.gauss(mu=mu,sigma=sigma)
    img[img>255] = 255
    img[img<0] = 0
    img = img.astype(np.uint8)
    return img

# 显示加噪后的图片


if __name__=='__main__':
    from matplotlib import pyplot as plt
    from skimage import io
    import skimage
    import pylab
    image=io.imread('../img/001.jpg')
    print(image)
    img=gauss_noise(image)
    # print(image)
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(image)
    plt.title("Origin picture")
    plt.subplot(122)
    plt.imshow(img)
    plt.title("Add Gaussian image")
    pylab.show()
