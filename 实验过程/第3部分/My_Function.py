#!/usr/bin/env python3

import numpy as np

def Build_Matrix(Second_Process, size):
    # 构建空矩阵
    Matrix = np.zeros((size,size), float)
    # 矩阵赋值到哪的光标
    i = 0
    # 将空矩阵进行赋值
    for line in range(size):
        for column in range(size):
            Matrix[line][column] = float(Second_Process[i])
            i += 1
    return Matrix

