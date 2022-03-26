#!/usr/bin/env python3

# 用于制作横竖训练集，第一项为标签，横为0，竖为1
import numpy as np
size = 3
# 竖
Matrix_list = []
for w in range(1, size):# 决定长度
    for z in range(0, w):# 决定位置
        for y in range(size):
            Matrix = np.zeros((size, size), int)
            for x in range(z, z+size-w+1):
                Matrix[x][y] = 1
            Matrix_list.append(Matrix)

with open('竖'+str(size)+'×'+str(size)+'.csv', 'w') as f:
    n = 1
    for ele in Matrix_list:
        if n == 1:
            f.write('1')
            n = 0
        else:
            f.write('\n1')
        for x in range(size):
            for y in range(size):
                f.write(',' + str(ele[x][y]))

# 横
Matrix_list = []
for w in range(1, size):# 决定长度
    for z in range(0, w):# 决定位置
        for x in range(size):
            Matrix = np.zeros((size, size), int)
            for y in range(z, z+size-w+1):
                Matrix[x][y] = 1
            Matrix_list.append(Matrix)

with open('横'+str(size)+'×'+str(size)+'.csv', 'w') as f:
    n = 1
    for ele in Matrix_list:
        if n == 1:
            f.write('0')
            n = 0
        else:
            f.write('\n0')
        for x in range(size):
            for y in range(size):
                f.write(',' + str(ele[x][y]))