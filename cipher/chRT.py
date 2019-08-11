# -*- coding: utf-8 -*-
'''
    @project:pag
    @file:chRT.py
    @ide:PyCharm
    @time:2019-07-29 01:23
    @author:Sun
    @todo:实现中国同余定理，如果除数列表有同样的数字的话，可能会出错，以下三个链接的算法都一样的实现方法
    @ref:以下用相同的算法实现
    https://www.geeksforgeeks.org/chinese-remainder-theorem-set-2-implementation/
    https://pypi.org/project/modint/
    https://rosettacode.org/wiki/Chinese_remainder_theorem#Python_3.6
'''


# todo 实现1
# A Python 3program to demonstrate
# working of Chinise remainder
# Theorem

# Returns modulo inverse of a with
# respect to m using extended
# Euclid Algorithm. Refer below
# post for details:
# https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m/
def inv(a, m):
    m0 = m
    x0 = 0
    x1 = 1
    if (m == 1):
        return 0
    # Apply extended Euclid Algorithm
    while (a > 1):
        # q is quotient
        q = a // m
        t = m
        # m is remainder now, process
        # same as euclid's algo
        m = a % m
        a = t
        t = x0
        x0 = x1 - q * x0
        x1 = t
        # Make x1 positive
    if (x1 < 0):
        x1 = x1 + m0
    return x1


# k is size of num[] and rem[].
# Returns the smallest
# number x such that:
# x % num[0] = rem[0],
# x % num[1] = rem[1],
# ..................
# x % num[k-2] = rem[k-1]
# Assumption: Numbers in num[]
# are pairwise coprime
# (gcd for every pair is 1)
def findMinX(num, rem):
    # Compute product of all numbers
    prod = 1
    for i in range(len(num)):
        prod = prod * num[i]

        # Initialize result
    result = 0

    # Apply above formula
    for i in range(len(num)):
        pp = prod // num[i]
        result = result + rem[i] * inv(pp, num[i]) * pp

    return result % prod


# TODO 实现2 https://rosettacode.org/wiki/Chinese_remainder_theorem#Python_3.6
from functools import reduce


def chinese_remainder(n, a): #n是模数，a是被模数
    sum = 0
    prod = reduce(lambda a, b: a * b, n)
    for n_i, a_i in zip(n, a):
        p = prod // n_i
        sum += a_i * mul_inv(p, n_i) * p
    return sum % prod


def mul_inv(a, b):
    b0 = b
    x0, x1 = 0, 1
    if b == 1: return 1
    while a > 1:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0: x1 += b0
    return x1


def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m


def ny(a, m):
    if a < 0:
        a %= m
        a += m
    for i in range(1, m):
        t = a * i
        if t % m == 1:
            return i


def getinvmodmat(m, a):  # 求矩阵m的模逆矩阵 https://blog.csdn.net/qq_37107465/article/details/84633130
    import numpy as np
    from decimal import getcontext
    getcontext().prec=50
    m_det = np.round(np.linalg.det(m))
    # print('行列式：\n', m_det)
    m_inv = np.linalg.inv(m)
    # print('逆矩阵：\n', m_inv)
    m_com = np.round(m_det * m_inv)  # 伴随矩阵
    # print('伴随矩阵：\n', m_com)
    detk_a = mul_inv(np.round(m_det % a), a)
    # detk_a=mul_inv(round(m_det%a),a)
    # print(detk_a)
    m_inv = np.round(detk_a * m_com % a)
    m_inv=m_inv.astype(int)
    # m_inv = round(detk_a * m_com) % a
    # print('模逆矩阵：\n', m_inv)
    return m_inv


def check_mat(A, M):  # 检查矩阵A是否存在Ｍ的模逆矩阵  https://www.zhihu.com/question/32675322
    from math import gcd
    import numpy as np
    det = int(np.round(np.linalg.det(A))) % M
    if np.gcd(det, M) > 1:
        return False
    return True


if __name__ == '__main__':
    n = [3, 5, 7]
    a = [2, 3, 2]
    print(chinese_remainder(n, a))
    # import numpy as np
    #
    # f=[15,14]
    # a=[0,11]
    # res=chinese_remainder(a,f)
    # print(res)

    # k = np.mat([[17, 44, 169, 126], [91, 121, 84, 85], [85, 71, 119, 25], [0, 85, 201, 44]],dtype=int)
    # print('原矩阵', k)
    # invk = getinvmodmat(k, 210)
    # a = (invk * k) % 210
    # print(a)

    # k = np.mat([[1,6,3],[4,7,9],[5,8,6]])
    # invk = getinvmodmat(k, 26)
    # a = (k * invk) % 26
    # print(a)
