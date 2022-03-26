#!/usr/bin/env python3

import numpy as np

# 将每行数据还原成矩阵的函数
def Build_Picture(Picture_List):
    # 将数据按逗号分开作为第二次处理
    Second_Process = Picture_List.split(',')
    # 记录下标签
    Lable = int(Second_Process[0])
    # 构建28X28的空矩阵
    Picture_Matrix = np.zeros((28,28), float)
    # 矩阵赋值到哪的光标
    i = 1
    # 将空矩阵进行赋值
    for line in range(28):
        for column in range(28):
            Picture_Matrix[line][column] = int(Second_Process[i]) / 255
            i += 1
    return Picture_Matrix, Lable

# 将矩阵进行二维180°旋转
def Flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr