#

a=[0.1,0.2]

a=map(lambda x:pow(x,2),a)
print(list(a))

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