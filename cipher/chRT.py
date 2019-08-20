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

import numpy as np

# TODO 实现2 https://rosettacode.org/wiki/Chinese_remainder_theorem#Python_3.6
from functools import reduce


def chinese_remainder(n, a):  # n是模数，a是被模数
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


def check_mat(A, M):  # 检查矩阵A是否存在Ｍ的模逆矩阵  https://www.zhihu.com/question/32675322
    from math import gcd
    import numpy as np
    det = int(np.round(np.linalg.det(A))) % M
    if np.gcd(det, M) > 1:
        return False
    return True


def getinvmodmat(m, a):  # 求矩阵m的模逆矩阵 https://blog.csdn.net/qq_37107465/article/details/84633130
    import numpy as np
    from decimal import getcontext
    getcontext().prec = 50
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
    m_inv = m_inv.astype(int)
    # m_inv = round(detk_a * m_com) % a
    # print('模逆矩阵：\n', m_inv)
    return m_inv


def getinvmodmat2(A, M):
    """
    :param A: 矩阵
    :param M: 模
    :return: 模逆矩阵
    """
    N = len(A)
    E = np.eye(N, dtype=int)  # 单位矩阵
    # print(E)
    A = np.hstack((A, E))  # 合并
    print(A)
    for j in range(N):
        for i in range(j, N):
            if np.gcd(A.item(i, j), M) > 1:
                print(np.gcd(A.item(i, j), M))
                continue
            tmp = mul_inv(A.item(i, j), M)
            print(j, i, tmp)
            A[i] *= tmp
            A[i] %= M
            if i != j:
                A[[i, j], :] = A[[j, i], :]
            break
        for i in range(N):
            if i != j:
                # A[i] -= (A.item(i, j) * A[j] % M)
                A[i] -= (A.item(i, j) * A[j])
                # for i in A[i]:
                #     if i.any()<0:
                #         i+=M
                A[i] %= M
        print(A)
    return A[:, N:N * 2]


def setK(N):  # c产生一个模N的4阶矩阵
    while True:
        K = np.mat(np.random.randint(0, N - 1, size=(4, 4)))  # todo
        if check_mat(K, N):
            return K


if __name__ == '__main__':
    K = np.mat([[17, 44, 169, 126], [91, 121, 84, 85], [13, 71, 119, 25], [0, 85, 201, 44]])
    # K[1]*=2
    # K[[0,1],:]=K[[1,0],:]
    print(K)
    a = getinvmodmat2(K, 30030)
    print(a)
    res = np.dot(K, a) % 30030
    print(res)

    # n = [3, 5, 7]
    # a = [2, 3, 2]
    # print(chinese_remainder(n, a))
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
