# -*- coding: utf-8 -*-
'''
    @project:pag
    @file:cryption.py
    @ide:PyCharm
    @time:2019-07-28 21:13
    @author:Sun
    @说明：TODO 直接使用了固定的参数作为加密用的参数，剩余的以后实现
'''

import random
import numpy as np
import math
from cipher.chRT import matmultimod


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

        # self.__K = np.mat([[17, 44, 169, 126], [91, 121, 84, 85], [85, 71, 119, 25], [0, 85, 201, 44]])
        # self.__invK = np.mat([[35, 31, 29, 0], [44, 57, 113, 29], [74, 157, 37, 194], [59, 27, 152, 103]])
        self.__K = None
        self.__invK = None
        self.__DIG = None
        self.__Cipher = None  # 密文
        self.__Real = None  # 明文

        # self.__setK()

    def Init(self):
        self.__setF()  # 得到F
        self.__setN()
        self.__setR()

    def __getPrimeNumber(self):
        """
        :param length:  素数的
        :return: 返回一个固定长度的素数
        """
        import cipher.bigprimenumber as pn
        # length = random.randint(0, self.__length)
        return pn.generate_prime_number(self.__length)

    def __setF(self):
        self.__F = []
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
        while True:
            self.__R = random.randint(0, self.__N - 1)
            from cipher.bigprimenumber import is_prime
            if is_prime(self.__R):
                return

    def __setK(self):
        from cipher.chRT import check_mat
        while True:
            self.__K = np.mat(np.random.randint(0, self.__N - 1, size=(4, 4)))  # todo
            if check_mat(self.__K, self.__N):
                return

    # 随机产生一个Zn的明文
    def gettext(self):
        return random.randint(0, self.__N - 1)

    def __setABC(self, x):
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

        from modint import chinese_remainder
        self.__a = chinese_remainder(self.__F, self.__A) % self.__N
        self.__b = chinese_remainder(self.__F, self.__B) % self.__N
        self.__c = chinese_remainder(self.__F, self.__C) % self.__N
        # print(self.__a, self.__b, self.__c)

        """from cipher.chRT import chinese_remainder
        self.__a = chinese_remainder(self.__F, self.__A) % self.__N
        self.__b = chinese_remainder(self.__F, self.__B) % self.__N
        self.__c = chinese_remainder(self.__F, self.__C) % self.__N
        print(self.__a, self.__b, self.__c)"""

    def reinit(self):
        # self.__F = [15, 14]
        # self.__M = 2
        # self.__N = 210
        # self.__K = np.mat([[17, 44, 169, 126], [91, 121, 84, 85], [85, 71, 119, 25], [0, 85, 201, 44]])

        self.__F = [22,  # 2*11
                    21,  # 3*7
                    65,  # 5*13
                    323,  # 17*19
                    # 667,  # 23*29
                    # 4087  # 61*67
                    ]
        self.__M = 4
        self.__N = 9699690  # 6469693230  # 39642633030  # 9699690 #30030
        self.__K = np.mat([[17, 44, 169, 126], [91, 121, 84, 85], [13, 71, 119, 25], [0, 85, 201, 44]])

        # self.__F = [22, 21, 65]
        # self.__M = 3
        # self.__N = 30030
        # self.__K = np.mat([[17, 44, 169, 126], [91, 121, 84, 85], [13, 71, 119, 25], [0, 85, 201, 44]])

        from cipher.chRT import check_mat
        if check_mat(self.__K, self.__N) == False:
            print("不是模逆矩阵")
            return
        from cipher.chRT import getinvmodmat
        self.__invK = getinvmodmat(self.__K, self.__N)
        print('用函数求出来的模逆矩阵:\n', self.__invK)

        # self.__invK = np.mat([[4475432465, 6014392064, 6292653233, 2132793728],
        #                       [4582859262, 3649936299, 308271498, 4352652179],
        #                       [2504084804, 1093385008, 3006458719, 2095201248],
        #                       [2498598530, 452477715, 3020965468, 105941323]])

        print('K: ', self.__K)
        # print('invK: ', self.__invK)

        print(matmultimod(self.__K.tolist(), self.__invK.tolist(), 4, self.__N))

        # self.__F = [22, 21, 65, 323]
        # self.__M = 4
        # self.__N = 9699690
        # self.__K = np.mat([[17, 44, 169, 126], [91, 121, 84, 85], [13, 71, 119, 25], [0, 85, 201, 44]])
        # self.__invK = np.mat([[7821825515, 50436115502, 45163305053, 22722937994],
        #                       [52757810644, 9476597139, 21449587732, 2489170559],
        #                       [27936671984, 38533605830, 27914121499, 24732580746],
        #                       [43508310480, 8011389075, 27298722460, 11418062143]])
        # print(self.__invK * self.__K % self.__N)
        # self.__setK()
        self.__setR()
        print('R: ', self.__R)

    def REencryption(self, x):
        self.__setABC(x)
        # print(self.__A, self.__B, self.__C)
        self.__setabc(x)
        self.__A.clear()
        self.__B.clear()
        self.__C.clear()
        self.__DIG = np.mat(np.diag([x, self.__a, self.__b, self.__c]))  # 对角矩阵
        # print('DIG:\n', self.__DIG)
        # self.__Cipher = (((self.__invK * self.__DIG) % self.__N) * self.__K) % self.__N
        res = matmultimod(self.__invK.tolist(), self.__DIG.tolist(), 4, self.__N)
        res = matmultimod(res, self.__K.tolist(), 4, self.__N)
        return res

        # return ((((self.__invK * self.__DIG) % self.__N) * self.__K) % self.__N)

    def REdecryption(self, cipher):
        # print('K:', self.__K)
        # print('invK:', self.__invK)
        for i in range(len(cipher)):
            for j in range(len(cipher[0])):
                cipher[i][j] %= self.__N

        res = matmultimod(self.__K.tolist(), cipher, 4, self.__N)
        self.__Real = matmultimod(res, self.__invK.tolist(), 4, self.__N)
        return self.__Real[0][0]

        # cipher = cipher % self.__N
        # np.mat(cipher)
        # self.__Real = np.round((((self.__K * cipher) % self.__N) * self.__invK) % self.__N)
        # return self.__Real.item(0, 0)
        # return self.__Real

    def encryption(self, x):

        self.__setABC(x)
        self.__setabc(x)
        from cipher.chRT import getinvmodmat
        self.__DIG = np.mat(np.diag([x, self.__a, self.__b, self.__c]))  # 对角矩阵
        # self.__invK = getinvmodmat(self.__K, self.__N)
        print((self.__K * self.__invK) % self.__N)
        # self.DIG = np.mat(np.diag([x, 147, 196, 91]))
        print('DIG:\n', self.__DIG)
        self.__Cipher = (self.__invK * self.__DIG * self.__K) % self.__N

    def decryption(self):
        self.__Real = np.round((self.__K * self.__Cipher * self.__invK.astype(int)) % self.__N)
        print('plain text:\n', self.__Real)
        self.__Real = self.__Real.item(0, 0)

    def getcip(self):
        return self.__Cipher

    def getPlaintext(self):
        return self.__Real

    def getN(self):
        return self.__N

    def getabc(self):
        return self.__a, self.__b, self.__c

    def changechiper(self, num, cipher):
        print('dig:', self.__DIG * (num % self.__N) % self.__N)
        cipher = cipher * (num % self.__N) % self.__N
        return cipher


if __name__ == '__main__':
    c = Cryption(M=2, length=5)
    c.reinit()
    x = random.randint(0, 255)
    # x=224
    print('plain:', x)
    tmp = c.REencryption(x)
    print('cipher:\n', tmp)
    # tmp = c.changechiper(122222, tmp)
    # print('cipher', tmp)
    tmp = c.REdecryption(tmp)
    print('plain:\n', tmp)
