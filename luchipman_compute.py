import decimal
from decimal import Decimal
from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
import cv2
import numpy as np

import argparse
import polanalyser as pa
import matplotlib.pyplot as plt
from numpy.linalg import svd

#计算穆勒矩阵
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="22test")
args = parser.parse_args()
path = args.input
print(f"Load images from '{path}'")
pcontainer = pa.PolarizationContainer(path)
image_list = pcontainer.get_list("image")

# 将Python列表转换为NumPy数组
array_image_list = np.array(image_list, dtype=np.float32)
print("图片数组大小：")
print(array_image_list.shape)
image_list = array_image_list.astype(np.float64)

mueller_psg_list = pcontainer.get_list("mueller_psg")
mueller_psa_list = pcontainer.get_list("mueller_psa")
img_mueller = pa.calcMueller(image_list, mueller_psg_list, mueller_psa_list)

# Normalized by m00 for visualization
img_mueller_normalized = img_mueller.copy()
img_mueller_normalized = img_mueller_normalized.astype(np.float64)
n1, n2 = img_mueller_normalized.shape[-2:]
img_mueller_normalized_m00 = img_mueller_normalized[..., 0, 0]
for i in range(n2):
    for j in range(n1):
        if i == 0 and j == 0:
            continue
        img_mueller_normalized[..., j, i] /= img_mueller_normalized_m00

img_mueller = img_mueller_normalized

#图片的穆勒矩阵元素，维度为图片(height，width)
m11_img = img_mueller[:, :, 0, 0]
m12_img = img_mueller[:, :, 0, 1]
m13_img = img_mueller[:, :, 0, 2]
m14_img = img_mueller[:, :, 0, 3]
m21_img = img_mueller[:, :, 1, 0]
m22_img = img_mueller[:, :, 1, 1]
m23_img = img_mueller[:, :, 1, 2]
m24_img = img_mueller[:, :, 1, 3]
m31_img = img_mueller[:, :, 2, 0]
m32_img = img_mueller[:, :, 2, 1]
m33_img = img_mueller[:, :, 2, 2]
m34_img = img_mueller[:, :, 2, 3]
m41_img = img_mueller[:, :, 3, 0]
m42_img = img_mueller[:, :, 3, 1]
m43_img = img_mueller[:, :, 3, 2]
m44_img = img_mueller[:, :, 3, 3]


import numpy as np
from math import fabs

#LU分解
def xzgetrf(A,ipiv):
    ipiv = [1, 2, 3, 4]
    info = 0

    for j in range(3):
        mmj_tmp = 2 - j
        b_tmp = j * 5
        jp1j = b_tmp + 2
        n = 4 - j
        jA = 0
        smax = fabs(A[b_tmp])

        for k in range(2, n + 1):
            s = fabs(A[b_tmp + k - 1])
            if s > smax:
                jA = k - 1
                smax = s

        if A[b_tmp + jA] != 0.0:
            if jA != 0:
                n = j + jA
                ipiv[j] = n + 1
                smax = A[j]
                A[j] = A[n]
                A[n] = smax
                smax = A[j + 4]
                A[j + 4] = A[n + 4]
                A[n + 4] = smax
                smax = A[j + 8]
                A[j + 8] = A[n + 8]
                A[n + 8] = smax
                smax = A[j + 12]
                A[j + 12] = A[n + 12]
                A[n + 12] = smax

            i = (b_tmp - j) + 4

            for n in range(jp1j, i + 1):
                A[n - 1] /= A[b_tmp]

        else:
            info = j + 1

        jA = b_tmp

        for jp1j in range(mmj_tmp + 1):
            n = (b_tmp + (jp1j << 2)) + 4
            smax = A[n]
            if A[n] != 0.0:
                i = jA + 6
                n = (jA - j) + 8
                for k in range(i, n + 1):
                    A[k - 1] += A[((b_tmp + k) - jA) - 5] * -smax
            jA += 4

    if info == 0 and not (A[15] != 0.0):
        info = 4

    return A,ipiv,info

"""
A, ipiv, info = xzgetrf(A)
print("A after LU decomposition:")
print(A)
print("Permutation vector:")
print(ipiv)
print("Info value:")
print(info)
"""
#SVD调用的函数：xrot,xrotg,xswap
def xrot(x, ix0, iy0, c, s):
    temp = x[iy0 - 1]
    temp_tmp = x[ix0 - 1]
    x[iy0 - 1] = c * temp - s * temp_tmp
    x[ix0 - 1] = c * temp_tmp + s * temp
    temp = c * x[ix0] + s * x[iy0]
    x[iy0] = c * x[iy0] - s * x[ix0]
    x[ix0] = temp
    temp = x[iy0 + 1]
    temp_tmp = x[ix0 + 1]
    x[iy0 + 1] = c * temp - s * temp_tmp
    x[ix0 + 1] = c * temp_tmp + s * temp
def xswap(x, ix0, iy0):
    temp = x[ix0 - 1]
    x[ix0 - 1] = x[iy0 - 1]
    x[iy0 - 1] = temp
    temp = x[ix0]
    x[ix0] = x[iy0]
    x[iy0] = temp
    temp = x[ix0 + 1]
    x[ix0 + 1] = x[iy0 + 1]
    x[iy0 + 1] = temp
import math

def xrotg(a, b):
    absa = abs(a)
    absb = abs(b)
    roe = b if absa <= absb else a
    scale = absa + absb

    if scale == 0.0:
        c = 1.0
        s = 0.0
        r = 0.0
        z = 0.0
    else:
        ads = absa / scale
        bds = absb / scale
        rad = math.sqrt(ads * ads + bds * bds)
        if roe < 0.0:
            rad = -rad
        c = absa / rad
        s = absb / rad
        r = scale * rad
        z = 1.0 if absa > absb else (1.0 / c if c != 0.0 else 1.0)

    return c, s, r, z

def b_xaxpy(n, a, x, ix0, y, iy0):
    if a != 0.0:
        for k in range(n):
            y[iy0 + k - 1] += a * x[ix0 + k - 1]

def c_xaxpy(n, a, x, ix0, y, iy0):
    if a != 0.0:
        for k in range(n):
            y[iy0 + k - 1] += a * x[ix0 + k - 1]

def xaxpy(n, a, ix0, y, iy0):
    if a != 0.0:
        for k in range(n):
            y[iy0 + k - 1] += a * y[ix0 + k - 1]

#奇异值分解SVD
import numpy as np

def svd(A):
    b_A = np.zeros(9)
    b_s = np.zeros(3)
    e = np.zeros(3)
    work = np.zeros(3)
    nrm = 0.0
    rt = 0.0
    scale = 0.0
    sm = 0.0
    snorm = 0.0
    sqds = 0.0
    ztest = 0.0
    exitg1 = 0
    k = 0
    kase = 0
    m = 0
    qq = 0
    U = np.zeros(9,)
    V = np.zeros(9,)

    for qjj in range(9):
        U[qjj] = 0.0
        V[qjj] = 0.0

    for q in range(2):
        qp1 = q + 1
        qq = q + 3 * q
        apply_transform = False
        nrm = np.linalg.norm(b_A[qq:])
        if nrm > 0.0:
            apply_transform = True
            if b_A[qq] < 0.0:
                ztest = -nrm
                b_s[q] = -nrm
            else:
                ztest = nrm
                b_s[q] = nrm
            if abs(ztest) >= 1.0020841800044864E-292:
                nrm = 1.0 / ztest
                b_A[qq:] *= nrm
            else:
                b_A[qq:] /= b_s[q]
            b_A[qq] += 1.0
            b_s[q] = -b_s[q]
        else:
            b_s[q] = 0.0
        for qjj in range(9):
            U[qjj] = 0.0
            V[qjj] = 0.0

        for q in range(2):
            qp1 = q + 1
            qq = q + 3 * q
            apply_transform = False
            nrm = np.linalg.norm(b_A[qq:])
            if nrm > 0.0:
                apply_transform = True
                if b_A[qq] < 0.0:
                    ztest = -nrm
                    b_s[q] = -nrm
                else:
                    ztest = nrm
                    b_s[q] = nrm
                if abs(ztest) >= 1.0020841800044864E-292:
                    nrm = 1.0 / ztest
                    b_A[qq:] *= nrm
                else:
                    b_A[qq:] /= b_s[q]
                b_A[qq] += 1.0
                b_s[q] = -b_s[q]
            else:
                b_s[q] = 0.0

            for kase in range(qp1, 4):
                qjj = q + 3 * (kase - 1)
                if apply_transform:
                    xdotc = np.dot(b_A[qq:], b_A[qjj + 1:qjj + 1 + 3 - q])
                    xaxpy = -(xdotc / b_A[qq])
                    b_A[qjj + 1:qjj + 1 + 3 - q] += xaxpy * b_A[qq + 1:qq + 1 + 3 - q]
                e[kase - 1] = b_A[qjj]

            for k in range(q + 1, 4):
                kase = (k + 3 * q) - 1
                U[kase] = b_A[kase]

            if qp1 <= 1:
                nrm = np.linalg.norm(e[1:])
                if nrm == 0.0:
                    e[0] = 0.0
                else:
                    if e[1] < 0.0:
                        e[0] = -nrm
                    else:
                        e[0] = nrm
                    nrm = e[0]
                    if abs(e[0]) >= 1.0020841800044864E-292:
                        nrm = 1.0 / e[0]
                        e[1:] *= nrm
                    else:
                        e[1:] /= nrm
                    e[1] += 1.0
                    e[0] = -e[0]
                    for k in range(qp1, 4):
                        work[k-1] = 0.0
                    for kase in range(qp1, 4):
                        work[kase - 1] = np.dot(e[kase - 1], b_A[3 * (kase - 1) + 1:3 * (kase - 1) + 1 + 2])
                    for kase in range(qp1, 4):
                        xaxpy = -e[kase - 1] / e[1]
                        b_A[3 * (kase - 1) + 1:3 * (kase - 1) + 1 + 2] += xaxpy * work[qp1 - 1:]
                for k in range(qp1, 4):
                    V[k-1] = e[k-1]

        m = 1
        b_s[2] = b_A[8]
        e[1] = b_A[7]
        e[2] = 0.0
        U[6] = 0.0
        U[7] = 0.0
        U[8] = 1.0

        for q in range(1, -1, -1):
            qp1 = q + 1
            qq = q + 3 * q
            if b_s[q] != 0.0:
                for kase in range(qp1, 4):
                    qjj = q + 3 * (kase - 1)
                    xdotc = np.dot(U[qq + 1:], U[qjj + 1:qjj + 1 + 3 - q])
                    xaxpy = -(xdotc / U[qq])
                    U[qjj + 1:qjj + 1 + 3 - q] += xaxpy * U[qq + 1:qq + 1 + 3 - q]
                for k in range(q + 1, 4):
                    kase = (k + 3 * q) - 1
                    U[kase] = -U[kase]
                U[qq] += 1.0
                if 0 <= q - 1:
                    U[3 * q] = 0.0
            else:
                U[3 * q] = 0.0
                U[3 * q + 1] = 0.0
                U[3 * q + 2] = 0.0
                U[qq] = 1.0

        for q in range(2, -1, -1):
            if (q + 1 <= 1) and (e[0] != 0.0):
                xdotc1 = np.dot(V[2:], V[5:])
                xaxpy1 = -(xdotc1 / V[1])
                V[5:] += xaxpy1 * V[2:]
                xdotc2 = np.dot(V[2:], V[8:])
                xaxpy2 = -(xdotc2 / V[1])
                V[8:] += xaxpy2 * V[2:]
            V[3 * q] = 0.0
            V[3 * q + 1] = 0.0
            V[3 * q + 2] = 0.0
            V[q + 3 * q] = 1.0
        qp1 = 0
        snorm = 0.0
        for q in range(1, -1, -1):
            qp1 = q + 1
            qq = q + 3 * q
            if b_s[q] != 0.0:
                for kase in range(qp1, 3):
                    qjj = q * 3
                    xaxpy = -(np.dot(U[qq:], U[qjj + 1:qjj + 1 + 3]) / U[qq])
                    U[qjj + 1:qjj + 1 + 3] *= (b_s[q] / abs(b_s[q]))
                b_s[q] *= (b_s[q] / abs(b_s[q]))
            if qp1 < 3 and e[q] != 0.0:
                for kase in range(qp1, 3):
                    qjj = (q + 1) * 3
                    xaxpy = -(np.dot(V[qp1 * 3:], V[qjj + 1:qjj + 1 + 3]) / V[qp1 * 3])
                    V[qjj + 1:qjj + 1 + 3] *= (e[q] / abs(e[q]))
                e[q] *= (e[q] / abs(e[q]))

        snorm = max(max(abs(b_s)), max(abs(e)))
    while m + 2 > 0 and qp1 < 75:
        k = m
        while True:
            q = k + 1
            if k + 1 == 0:
                break
            else:
                nrm = abs(e[k])
                if (nrm <= 2.2204460492503131E-16 * (abs(b_s[k]) + abs(b_s[k + 1]))) or \
                        (nrm <= 1.0020841800044864E-292) or \
                        ((qp1 > 20) and (nrm <= 2.2204460492503131E-16 * snorm)):
                    e[k] = 0.0
                    break
                else:
                    k -= 1

        if k + 1 == m + 1:
            kase = 4
        else:
            qjj = m + 2
            kase = m + 2
            exitg2 = False
            while kase >= k + 1:
                qjj = kase
                if kase == k + 1:
                    exitg2 = True
                else:
                    nrm = 0.0
                    if kase < m + 2:
                        nrm = abs(e[kase - 1])
                    if kase > k + 2:
                        nrm += abs(e[kase - 2])
                    ztest = abs(b_s[kase - 1])
                    if (ztest <= 2.2204460492503131E-16 * nrm) or \
                            (ztest <= 1.0020841800044864E-292):
                        b_s[kase - 1] = 0.0
                        exitg2 = True
                    else:
                        kase -= 1
            if qjj == k + 1:
                kase = 3
            elif qjj == m + 2:
                kase = 1
            else:
                kase = 2
                q = qjj
        if kase == 1:
            ztest = e[m]
            e[m] = 0.0
            qjj = m + 1
            for k in range(qjj, q, -1):
                xrotg(b_s[k - 1],ztest,sm,sqds)
                if k > q + 1:
                    ztest = -sqds * e[0]
                    e[0] *= sm
                xrot(V, 3 * (k - 1) + 1, 3 * (m + 1) + 1, sm, sqds)
            break
        elif kase == 2:
            ztest = e[q - 1]
            e[q - 1] = 0.0
            for k in range(q + 1, m + 2):
                xrotg(b_s[k - 1],ztest,sm,sqds)
                rt = e[k - 1]
                ztest = -sqds * rt
                e[k - 1] = rt * sm
                xrot(U, 3 * (k - 1) + 1, 3 * (q - 1) + 1, sm, sqds)
            break
        elif kase == 3:
            kase = m + 1
            nrm = b_s[m + 1]
            scale = max(max(max(max(abs(nrm), abs(b_s[m])), abs(e[m])), abs(b_s[q])), abs(e[q]))
            sm = nrm / scale
            nrm = b_s[m] / scale
            ztest = e[m] / scale
            sqds = b_s[q] / scale
            rt = ((nrm + sm) * (nrm - sm) + ztest * ztest) / 2.0
            nrm = sm * ztest
            nrm *= nrm
            if (rt != 0.0) or (nrm != 0.0):
                ztest = np.sqrt(rt * rt + nrm)
                if rt < 0.0:
                    ztest = -ztest
                ztest = nrm / (rt + ztest)
            else:
                ztest = 0.0
            ztest += (sqds + sm) * (sqds - sm)
            nrm = sqds * (e[q] / scale)
            for k in range(q + 1, kase + 1):
                xrotg(ztest, nrm, sm, sqds)
                if k > q + 1:
                    e[0] = ztest
                nrm = e[k - 1]
                rt = b_s[k - 1]
                e[k - 1] = sm * nrm - sqds * rt
                ztest = sqds * b_s[k]
                b_s[k] *= sm
                xrot(V, 3 * (k - 1) + 1, 3 * k + 1, sm, sqds)
                b_s[k - 1] = sm * rt + sqds * nrm
                xrotg(b_s[k - 1], ztest, sm, sqds)
                ztest = sm * e[k - 1] + sqds * b_s[k]
                b_s[k] = -sqds * e[k - 1] + sm * b_s[k]
                nrm = sqds * e[k]
                e[k] *= sm
                xrot(U, 3 * (k - 1) + 1, 3 * k + 1, sm, sqds)
            e[m] = ztest
            qp1 += 1
            break
        else:
            if b_s[q] < 0.0:
                b_s[q] = -b_s[q]
                kase = 3 * q
                qjj = kase + 3
                for k in range(kase + 1, qjj + 1):
                    V[k - 1] = -V[k - 1]
            qp1 = q + 1
            while (q + 1 < 3) and (b_s[q] < b_s[qp1]):
                rt = b_s[q]
                b_s[q] = b_s[qp1]
                b_s[qp1] = rt
                xswap(V, 3 * q + 1, 3 * (q + 1) + 1)
                xswap(U, 3 * q + 1, 3 * (q + 1) + 1)
                q = qp1
                qp1 += 1
            qp1 = 0
            m -= 1
            break
    return U, b_s, V


#计算三个极化分解矩阵
#计算点（x,y）的双衰减矩阵MD
def compute_MM_polarLuChipman(x, y):
    #输入的穆勒矩阵
    m11 = m11_img[x][y]
    m12 = m12_img[x][y]
    m13 = m13_img[x][y]
    m14 = m14_img[x][y]
    m21 = m21_img[x][y]
    m22 = m22_img[x][y]
    m23 = m23_img[x][y]
    m24 = m24_img[x][y]
    m31 = m31_img[x][y]
    m32 = m32_img[x][y]
    m33 = m33_img[x][y]
    m34 = m34_img[x][y]
    m41 = m41_img[x][y]
    m42 = m42_img[x][y]
    m43 = m43_img[x][y]
    m44 = m44_img[x][y]

    b = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int8)#3x3单位矩阵
    M = np.zeros((16,))
    MD = np.zeros((16,))
    M_0 = np.zeros((16,))
    Mdelta = np.zeros((16,))
    U = np.zeros((9,))
    V = np.zeros((9,))
    ipiv = np.zeros((4,), dtype=np.int8)
    b_D1 = np.zeros((9,))
    dvec = [0] * 3

    #按行拼接穆勒矩阵元素
    M[0] = m11
    M[4] = m12
    M[8] = m13
    M[12] = m14
    M[1] = m21
    M[5] = m22
    M[9] = m23
    M[13] = m24
    M[2] = m31
    M[6] = m32
    M[10] = m33
    M[14] = m34
    M[3] = m41
    M[7] = m42
    M[11] = m43
    M[15] = m44

    #双衰减矩阵MD
    D = 1.0 / m11
    dvec[0] = m12 * D
    dvec[1] = m13 * D
    dvec[2] = m14 * D
    D = np.sqrt((m12 * m12 + m13 * m13) + m14 * m14)
    D1_tmp = D * D

    D1 = np.sqrt(1.0 - D1_tmp)
    if D == 0.0:#D=0 时，无法进行LU分解
        for i in range (16):
            M_0[i] = M[i]
            MD[i] = 0.0
        MD[0] = 1.0
        MD[5] = 1.0
        MD[10] = 1.0
        MD[15] = 1.0
    else:
        MD[0] = 1.0
        for i in range(3):
            MD_tmp = (i + 1) << 2 #三次循环分别是4,8,12
            d = dvec[i] #1/m12,1/m13,1/m14
            MD[MD_tmp] = d #为MD[4][8][12],即MD21,31,41赋值
            MD[i + 1] = d #为MD[1][2][3],即MD12,13,14赋值
            MD[MD_tmp + 1] = D1 * b[3 * i] + (1.0 - D1) * dvec[0] * dvec[i] / D1_tmp #计算mD,即MD右下角3x3区域
            MD[MD_tmp + 2] = D1 * b[3 * i + 1] + (1.0 - D1) * dvec[1] * dvec[i] / D1_tmp
            MD[MD_tmp + 3] = D1 * b[3 * i + 2] + (1.0 - D1) * dvec[2] * dvec[i] / D1_tmp

        # 复制 M 和 MD 的值到 M_0 和 Mdelta
        M_0 = np.copy(M)
        Mdelta = np.copy(MD)
        #LU分解
        Mdelta,ipiv,info = xzgetrf(Mdelta,ipiv)

        # 循环迭代计算 M_0 的值
        for j in range(4):#0,1,2,3（第一行）
            MD_tmp = j << 2#0,4,8,12
            for k in range(j):
                kBcol = k << 2#0,4,8,12（第一列）
                d = Mdelta[k + MD_tmp]#0,5,10,15（对角线）
                if d != 0.0:
                    M_0[MD_tmp] -= d * M_0[kBcol]
                    M_0[MD_tmp + 1] -= d * M_0[kBcol + 1]
                    M_0[MD_tmp + 2] -= d * M_0[kBcol + 2]
                    M_0[MD_tmp + 3] -= d * M_0[kBcol + 3]

            D = 1.0 / Mdelta[j + MD_tmp]#0,5,10,15(斜对角线）

            M_0[MD_tmp] *= D#0,1,2,3（第一行）
            M_0[MD_tmp + 1] *= D
            M_0[MD_tmp + 2] *= D
            M_0[MD_tmp + 3] *= D

        #print("M_0(1)")
        #print(M_0)

        for j in range(3, -1, -1):#3,2,1,0
            MD_tmp = (j << 2) - 1#11,7,3,-1
            i = j + 2#13,9,5,1
            for k in range(i, 5):
                kBcol = (k - 1) << 2
                d = Mdelta[k + MD_tmp]
                if d != 0.0:
                    M_0[MD_tmp + 1] -= d * M_0[kBcol]
                    M_0[MD_tmp + 2] -= d * M_0[kBcol + 1]
                    M_0[MD_tmp + 3] -= d * M_0[kBcol + 2]
                    M_0[MD_tmp + 4] -= d * M_0[kBcol + 3]

        #print("M_0(2)")
        #print(M_0)

        for j in range(2, -1, -1):#2,1,0
            i = ipiv[j]#0,0,0
            if i != j + 1:
                k = j * 4#8,4,0
                D = M_0[k]
                MD_tmp = (i - 1) * 4
                M_0[k] = M_0[MD_tmp]
                M_0[MD_tmp] = D
                D = M_0[k + 1]
                M_0[k + 1] = M_0[MD_tmp + 1]
                M_0[MD_tmp + 1] = D
                D = M_0[k + 2]
                M_0[k + 2] = M_0[MD_tmp + 2]
                M_0[MD_tmp + 2] = D
                D = M_0[k + 3]
                M_0[k + 3] = M_0[MD_tmp + 3]
                M_0[MD_tmp + 3] = D

        #print("M_0(3):")
        #print(M_0)

    #计算相位延迟矩阵MR
    isodd = True
    for k in range(9):
        if isodd:
            D = M_0[(k % 3 + ((k // 3 + 1) << 2)) + 1]
            if np.isinf(D) or np.isnan(D):
                isodd = False
        else:
            isodd = False

    if isodd:
        for i in range(3):#0,1,2
            MD_tmp = (i + 1) << 2#4,8,12
            b_D1[3 * i] = M_0[MD_tmp + 1]
            b_D1[3 * i + 1] = M_0[MD_tmp + 2]
            b_D1[3 * i + 2] = M_0[MD_tmp + 3]
        b_D1 = b_D1.reshape(3,3)
        V, dvec, U = np.linalg.svd(b_D1)

        #消除linalg.svd与c函数svd的结果区别
        U[0, :] = -U[0, :]
        U[2, :] = -U[2, :]
        V = V.T
        V[0, :] = -V[0, :]
        V[2, :] = -V[2, :]

    else:
        U = np.full((9,), np.nan)
        V = np.full((9,), np.nan)

    U = U.reshape(9, )
    V = V.reshape(9, )
    b_D1 = b_D1.reshape(9, )

    M, ipiv, info = xzgetrf(M, ipiv)

    #print("MR part ipiv value:")
    #print(ipiv)

    isodd = ipiv[0] > 1
    if ipiv[1] > 2:
        isodd = not isodd

    D = M[0] * M[5] * M[10] * M[15]

    if ipiv[2] > 3:
        isodd = not isodd

    if isodd:
        D = -D

    if D < 0.0:
        dvec[2] = -1.0
    else:
        dvec[2] = 1.0

    b_b = np.zeros((9,))
    b_b[0] = 1
    b_b[4] = 1
    b_b[8] = int(dvec[2])

    for i in range(3):
        d = U[i]
        D = U[i + 3]
        D1_tmp = U[i + 6]
        for MD_tmp in range(3):
            b_D1[i + 3 * MD_tmp] = (d * b_b[3 * MD_tmp] + D * b_b[3 * MD_tmp + 1]) + D1_tmp * b_b[3 * MD_tmp + 2]
        d = b_D1[i]
        D = b_D1[i + 3]
        D1_tmp = b_D1[i + 6]
        for MD_tmp in range(3):
            U[i + 3 * MD_tmp] = (d * V[MD_tmp] + D * V[MD_tmp + 3]) + D1_tmp * V[MD_tmp + 6]
    #print("b_D1(2):")
    #print(b_D1)

    M[0] = 1.0
    M[4] = 0.0
    M[8] = 0.0
    M[12] = 0.0
    for i in range(3):
        M[i + 1] = 0.0
        MD_tmp = (i + 1) << 2
        M[MD_tmp + 1] = U[3 * i]
        M[MD_tmp + 2] = U[3 * i + 1]
        M[MD_tmp + 3] = U[3 * i + 2]
    #python与c不同的部分
    M=M.T

    #退偏矩阵
    for i in range(4):
        d = M_0[i]
        D = M_0[i + 4]
        D1_tmp = M_0[i + 8]
        D1 = M_0[i + 12]
        for MD_tmp in range(4):
            Mdelta[i + (MD_tmp << 2)] = ((d * M[MD_tmp] + D * M[MD_tmp + 4]) + D1_tmp * M[MD_tmp + 8]) + D1 * M[
                MD_tmp + 12]
    # python与c不同的部分
    Mdelta = Mdelta.T
    return MD,M,Mdelta


def rt_atan2d_snf(u0, u1):
    if math.isnan(u0) or math.isnan(u1):
        y = float('nan')
    elif math.isinf(u0) and math.isinf(u1):
        b_u0 = 1 if u0 > 0.0 else -1
        b_u1 = 1 if u1 > 0.0 else -1
        y = math.atan2(b_u0, b_u1)
    elif u1 == 0.0:
        if u0 > 0.0:
            y = math.pi / 2.0
        elif u0 < 0.0:
            y = -(math.pi / 2.0)
        else:
            y = 0.0
    else:
        y = math.atan2(u0, u1)

    return y

import math
import numpy as np

def compute_Retard_Params(i,j):
    M = compute_MM_polarLuChipman(i, j)[1]
    M = M.astype(np.float64)

    p11 = M[0]
    p12 = M[1]
    p13 = M[2]
    p14 = M[3]
    p21 = M[4]
    p22 = M[5]
    p23 = M[6]
    p24 = M[7]
    p31 = M[8]
    p32 = M[9]
    p33 = M[10]
    p34 = M[11]
    p41 = M[12]
    p42 = M[13]
    p43 = M[14]
    p44 = M[15]
    MR = np.array([[p11, p12, p13, p14],
                   [p21, p22, p23, p24],
                   [p31, p32, p33, p34],
                   [p41, p42, p43, p44]])

    MRC = np.zeros((4, 4))
    MRL = np.zeros((4, 4))

    argu_tmp = p22 + p33
    argu = 0.5 * (argu_tmp + p44) - 0.5
    if abs(argu) > 1.0:
        if argu > 0.0:
            d = 0.0
        else:
            d = math.pi
    else:
        d = math.acos(argu)

    #57.295779513082323 是 180/π 的近似值。将弧度转换为角度时，可以使用这个因子进行转换。
    totR = d * 180.0 / math.pi
    linR = 57.295779513082323 * math.acos(p44)
    cirR = 57.295779513082323 * math.atan2((p32 - p23), argu_tmp)
    argu = cirR / 2.0 * math.pi / 180.0
    argu_tmp = math.sin(2.0 * argu)
    argu = math.cos(2.0 * argu)

    MRC[1][1] = argu
    MRC[1][2] = -argu_tmp
    MRC[2][1] = argu_tmp
    MRC[2][2] = argu
    MRC[0][0] = 1.0
    MRC[3][3] = 1.0

    #argu = abs(cirR)
    if argu < 0.0001:
        MRL = MR
    else:
        for i in range(4):
            for j in range(4):
                MRL[i][j] = sum(MR[i][k] * MRC[k][j] for k in range(4))

        if abs(57.295779513082323 * math.atan2((MRL[1][2] - MRL[2][1]), (MRL[1][1] + MRL[2][2]))) > argu:
            for i in range(4):
                for j in range(4):
                    MRL[i][j] = sum(MR[i][k] * MRC[k][j] for k in range(4))

    oriR = 57.295779513082323 * math.atan2(MRL[3][2], MRL[3][0])
    if oriR < 0.0:
        oriR = 360.0 - abs(oriR)

    oriRfull = 0.5 * oriR
    oriR = oriRfull + 90.0

    return totR, linR, cirR, oriR, oriRfull


"""
def compute_Retard_Params(i,j):
    M = compute_MM_polarLuChipman(i, j)[1]
    p11 = M[0]
    p12 = M[1]
    p13 = M[2]
    p14 = M[3]
    p21 = M[4]
    p22 = M[5]
    p23 = M[6]
    p24 = M[7]
    p31 = M[8]
    p32 = M[9]
    p33 = M[10]
    p34 = M[11]
    p41 = M[12]
    p42 = M[13]
    p43 = M[14]
    p44 = M[15]
    R = math.acos((p11+p22+p33+p44)/2-1)
    return R
"""
def compute_Retard_Params_img():
    img_width = len(img_mueller)
    img_height = len(img_mueller[0])
    img_delta_array = np.zeros([img_width, img_height])
    for i in range(img_width):  # 宽
        for j in range(img_height):  # 高
            R = compute_Retard_Params(i, j)[0]
            img_delta_array[i, j] = R
    return img_delta_array

#计算双衰减系数（总双衰减totD,线性linD,圆双衰减cirD）
def compute_Diatt_Params_img():
    # 讨论mueller.shape == (height, width, 4, 4)和mueller.shape == (height, width, 4, 4)的情况
    if img_mueller.shape[2] == 4 and img_mueller.shape[3] == 4:
        m14_img = img_mueller[:, :, 0, 3]
        # Diattenuation
        totD_tmp = m12_img * m12_img + m13_img * m13_img
        totD = np.sqrt(totD_tmp + m14_img * m14_img) / m11_img
        linD = np.sqrt(totD_tmp)
        cirD = np.fabs(m14_img)
        #return totD, linD, cirD #调用：compute_Diatt_Params()[0],compute_Diatt_Params()[1],...
        return totD
    else:
        totD_tmp = m12_img * m12_img + m13_img * m13_img
        linD = np.sqrt(totD_tmp)
        return totD_tmp

#计算退偏系数
def compute_Delta_Params_img():
    # 讨论mueller.shape == (height, width, 4, 4)和mueller.shape == (height, width, 4, 4)的情况
    img_width = len(img_mueller)
    img_height = len(img_mueller[0])
    if img_mueller.shape[2] == 4 and img_mueller.shape[3] == 4:
        img_delta_array = np.zeros([img_width,img_height])
        for i in range(img_width):#宽
            for j in range(img_height):#高
                Mdelta = compute_MM_polarLuChipman(i,j)[2]
                d22 = Mdelta[5]
                d33 = Mdelta[10]
                d44 = Mdelta[15]
                delta = 1.0 - ((fabs(d22) + fabs(d33)) + fabs(d44)) / 3.0
                img_delta_array[i,j] = delta
        return img_delta_array
    else:
        print("only 4x4 matrix has Depolarisation Matrix")
        return 0

#绘制退偏图像
def plot_Depol(depol_img_array):
    # 使用OpenCV创建图像
    img_array = 100000 * depol_img_array

    image = img_array.astype(np.uint8)

    # 使用OpenCV的伪彩色映射函数将灰度图转换为伪彩色图
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    # 显示图像
    plt.imshow(image)
    plt.colorbar()
    plt.show()

#绘制延迟图像
def plot_Retard(retard_img_array):
    # 使用OpenCV创建图像
    img_array = 100000 * retard_img_array

    image = img_array.astype(np.uint8)

    image = cv2.convertScaleAbs(image, alpha=10)

    # 使用OpenCV的伪彩色映射函数将灰度图转换为伪彩色图
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    # 显示图像
    plt.imshow(image)
    plt.colorbar()
    plt.show()

def plot_Diatt(diatt_img_array):

    # 使用OpenCV创建图像
    img_array = 100000 * diatt_img_array

    # 获取第一轴的大小（元组中包含的子元组的个数）
    img_height = len(img_array)
    print("height:", img_height)

    # 获取第二轴的大小（元组中每个子元组的长度）
    img_width = len(img_array[0])
    print("width:", img_width)

    image = img_array.astype(np.uint8)

    # 使用OpenCV的伪彩色映射函数将灰度图转换为伪彩色图
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    # 显示图像
    plt.imshow(image)
    plt.colorbar()
    plt.show()

print(img_mueller[220, 220,:, :])
print("MD")
print(compute_MM_polarLuChipman(220, 220)[0].reshape(4,4))
print("MR")
print(compute_MM_polarLuChipman(220, 220)[1].reshape(4,4))
print("Mdelta")
print(compute_MM_polarLuChipman(220, 220)[2].reshape(4,4))

plot_Diatt(compute_Diatt_Params_img())

plot_Depol(compute_Delta_Params_img())

plot_Retard(compute_Retard_Params_img())
