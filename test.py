#
import numpy as np
a=22*21*65*323*667
print(a)

a = 17 * 3188752432 + 44 * 23781510238 + 169 * 11590684370 + 126 * 38958317798
print(a)
a %= 39642633030
print(a)
#
# a=91 * 7254113 + 121 * 7582091 + 84 * 9254509 + 85 * 4360904
# print(a)
# print(a%9699690)

# k=int(input('个数：'))
#
# res=1
# for i in range(k):
#     tmp=int(input(str(i)+'is:'))
#     res*=tmp
#
# print(res)
# from PIL import Image
# image=Image.open('/home/sun/Pictures/壁纸/002.jpeg')
#
# image=image.transpose(Image.FLIP_LEFT_RIGHT)
#
# image.save('iii.jpg')

# import numpy as np
# import random
#
# for i in range(100):
#     print(random.gauss(mu=0, sigma=60))

# a=[[i*j for i in range(5)] for j in range(5)]
# print(a)
# b=[i[1:-1] for i in a][1:-1]
# print(b)


# import numpy as np
#
# a=np.random.normal(0,0.5,[1,10])
# print(a)

# a=[0.1,0.2]
#
# a=map(lambda x:pow(x,2),a)
# print(list(a))

# def ge(a):
#     if a<4:
#         print('a')
#     elif a>=4 and a<8:
#         print('b')
#     else:
#         print('c')
# ge(5)

# class Test:
#     def __init__(self):
#         self.__d=5
#
#     def pri(self):
#         def p():
#             print(self.__d)
#
#         p()
#
# t=Test()
#
# t.pri()


# print('%2d-%02d' % (3, 1))
# print('%.2f' % 3.1415926)
#
# a=list(range(10))
# a=sum(a)
# print(a)

#
# import numpy as np
#
# def gray1(image):
#     print(len(image))
#     print(len(image[0]))
#     grayimage=[0]*len(image[0])
#     grayimage=[grayimage for i in range(len(image))]
#     print(len(grayimage))
#     print(len(grayimage[0]))
#
#
#     for i in range(len(image)):
#         for j in range(len(image[0])):
#             grayimage[i][j]=(0.229*image[i][j][0]+0.587*image[i][j][1]+0.114*image[i][j][2])
#
#     return np.array(grayimage)
#
# def gray2(image):
#     # print(len(image))
#     # print(len(image[0]))
#     # grayimage=[0]*len(image[0])
#     # grayimage=[grayimage]*len(image)
#     # print(len(grayimage))
#     # print(len(grayimage[0]))
#
#     grayimage=[]
#
#     for i in range(len(image)):
#         grayimage.append([])
#         for j in range(len(image[0])):
#             grayimage[i].append(0.229*image[i][j][0]+0.587*image[i][j][1]+0.114*image[i][j][2])
#
#     return np.array(grayimage)
#
# from skimage import io
# import matplotlib.pyplot as plt
#
# image=io.imread('img/001.jpg')
# print('+++++++++++++++++++++++++++++++++++++\n',image)
# grayimage1=gray1(image)
# plt.imshow(grayimage1)
# plt.show()
# print('-------------------------------------\n',grayimage1)
#
# grayimage2=gray2(image)
# plt.imshow(grayimage2)
# plt.show()
# print('*************************************\n',grayimage2)
