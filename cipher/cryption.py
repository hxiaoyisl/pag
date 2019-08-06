# -*- coding: utf-8 -*-
'''
    @project:pag
    @file:cryption.py
    @ide:PyCharm
    @time:2019-07-28 21:13
    @author:Sun
'''

import random
import numpy as np
import math
import cipher.matrix as mt


class Cryption:
    def __init__(self, M, length=1024):
        """
        :param M:
        :param length:  产生的大质数的长度
        """
        self.__length = length
        self.__M = M  #
        self.__F = []
        # self.__F=[15,14]
        self.__N = None

        self.__R = None

        self.__A = []
        self.__B = []
        self.__C = []

        self.__a = None
        self.__b = None
        self.__c = None

        self.__K = np.mat([[17, 44, 169, 126], [91, 121, 84, 85], [85, 71, 119, 25], [0, 85, 201, 44]])
        self.__invK = np.mat([[35, 31, 29, 0], [44, 57, 113, 29], [74, 157, 37, 194], [59, 27, 152, 103]])
        # self.__K =None
        # self.__invK=None
        self.__DIG = None
        self.__Cipher = None  # 密文
        self.__Real = None  # 明文

        self.__setF()  # 得到F
        self.__setN()
        self.__setR()
        # self.__setK()

    def __getPrimeNumber(self):
        """
        :param length:  素数的
        :return: 返回一个固定长度的素数
        """
        import cipher.bigprimenumber as pn
        # length = random.randint(0, self.__length)
        return pn.generate_prime_number(self.__length)

    def __setF(self):
        while self.__F.__len__() < self.__M:
            a = self.__getPrimeNumber()
            b = self.__getPrimeNumber()
            print(a, b)
            t = a * b
            label = True
            # for i in self.__F:
            #     if np.gcd(i, t) > 1:
            #         label = False
            #         break
            if label == True:
                self.__F.append(t)
            # print(a*b)

    def __setN(self):
        self.__N = 1
        for tf in self.__F:
            self.__N *= tf

    def __setR(self):
        self.__R = random.randint(0, self.__N-1)

    def __setK(self):
        from cipher.chRT import check_mat
        while True:
            self.__K = np.mat(np.random.randint(0, self.__N-1, size=(4, 4)))  # todo
            if check_mat(self.__K, self.__N):
                return

    # 随机产生一个Zn的明文
    def gettext(self):
        return random.randint(0, self.__N-1)

    def __setABC(self, __x):
        for i in range(self.__M):
            tmp = random.random()
            if tmp < self.__M / (self.__M + 1):
                self.__A.append(x)
                self.__B.append(self.__R)
                self.__C.append(self.__R)
            elif tmp >= self.__M / (self.__M + 1) and tmp < 1 - 1 / (2 * (self.__M + 1)):
                self.__A.append(self.__R)
                self.__B.append(x)
                self.__C.append(self.__R)
            else:
                self.__A.append(self.__R)
                self.__B.append(self.__R)
                self.__C.append(x)

    # 使用中国剩余定理产生a,b,c
    def __setabc(self, x):
        '''
        :param x:  明文
        :return:
        '''
        # self.__setABC(x)  # 初始化A，B，C
        import sys
        sys.setrecursionlimit(100000)  # ref:https://blog.csdn.net/cliviabao/article/details/79927186
        import cipher.CRT as crt
        """from modint import chinese_remainder
        self.__a = chinese_remainder(self.__A,self.__F)% self.__N
        self.__b = chinese_remainder(self.__B, self.__F)% self.__N
        self.__c = chinese_remainder(self.__C, self.__F)% self.__N"""
        self.__a = crt.crt(self.__A[:], self.__F[:]) % self.__N
        self.__b = crt.crt(self.__B[:], self.__F[:]) % self.__N
        self.__c = crt.crt(self.__C[:], self.__F[:]) % self.__N

    def encryption(self, x):
        # np.set_printoptions(suppress=True)
        # print(self.__K)
        from cipher.chRT import check_mat
        if check_mat(self.__K, self.__N) == False:
            print("不是模逆矩阵")
            return

        self.__setABC(x)
        self.__setabc(x)
        from cipher.chRT import getinvmodmat
        self.__DIG = np.mat(np.diag([x, self.__a, self.__b, self.__c]))  # 对角矩阵
        self.__invK = getinvmodmat(self.__K, self.__N)
        print((self.__K * self.__invK) % self.__N)
        # self.DIG = np.mat(np.diag([x, 147, 196, 91]))
        print('DIG:\n', self.__DIG)
        self.__Cipher = (self.__invK * self.__DIG * self.__K) % self.__N

    def decryption(self):
        self.__Real = np.round((self.__K * self.__Cipher * self.__invK.astype(int)) % self.__N)
        print('plain text:\n', self.__Real)
        self.__Real = self.__Real.item(0,0)

    def getcip(self):
        return self.__Cipher

    def getPlaintext(self):
        return self.__Real

    def getN(self):
        return self.__N

    def getabc(self):
        return self.__a, self.__b, self.__c


if __name__ == '__main__':
    c = Cryption(M=2, length=5)
    print('N:', c.getN())
    # print(c.K)
    x = c.gettext()
    print('明文：', x)
    c.encryption(x)
    print('cipher:\n', c.getcip())
    c.decryption()
    print(c.getPlaintext())
