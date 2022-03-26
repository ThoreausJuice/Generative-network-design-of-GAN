#!/usr/bin/env python3

# 单隐层卷积生成器

from My_Function import *
import numpy as np
from scipy import signal
import math

# 各层神经元输入
Neuron_Input = []
# 各层间权值矩阵
Weight = []
# 各层神经元输出
Neuron_Output = []
# 神经网络隐层层数设定
n = 1
# 图片大小设定
Picture_Size = 28
# 卷积核初始大小设定
Convolution_kernel_size = 3
# 学习率设定
η = 0.5
# 误差目标级数
level = 100
# 卷积核设定
for i in range(n+1):
    Weight.append(np.random.rand(Convolution_kernel_size, Convolution_kernel_size))
# 读取数据集
with open('0样本.csv', 'r') as f:
    Original_String = f.read()
First_Process = Original_String.split('\n')

# 底层随机噪音输出设定
Neuron_Output.append(np.random.rand(Picture_Size, Picture_Size))
# 目标图片设定
Target, Lable = Build_Picture(First_Process[0])

E_total = 1
while E_total > 1/level:
    # 正向计算
    # 输入层到隐层
    Flip_Weight = Flip180(Weight[0])
    Feature_map = signal.convolve2d(Neuron_Output[0], Flip_Weight, mode = 'same')
    Neuron_Input.append(Feature_map)
    # 隐层处理
    Neuron_Output.append(Feature_map)
    for i in range(Picture_Size):
        for j in range(Picture_Size):
            if Neuron_Input[0][i][j] < 0:
                Neuron_Output[1][i][j] = 0
    # 隐层到输出层
    Flip_Weight = Flip180(Weight[1])
    Feature_map = signal.convolve2d(Neuron_Output[1], Flip_Weight, mode = 'same')
    Neuron_Input.append(Feature_map)
    # 输出层处理
    Neuron_Output.append(np.zeros((Picture_Size, Picture_Size), float))
    for i in range(Picture_Size):
        for j in range(Picture_Size):
            Neuron_Output[2][i][j] = 1 / (1 + math.exp(-Neuron_Input[1][i][j]))

    # 计算误差
    Error_map = np.zeros((Picture_Size, Picture_Size), float)
    E_total = 0
    for i in range(Picture_Size):
        for j in range(Picture_Size):
            Error_map[i][j] = 0.5 * (Target[i][j] - Neuron_Output[2][i][j]) ** 2
            E_total += Error_map[i][j]


    # 反向计算
    # 计算并存储每层节点输出对输入的偏导
    Node_PD = []
    PD = np.zeros((Picture_Size, Picture_Size), float)
    for i in range(Picture_Size):
        for j in range(Picture_Size):
            PD[i][j] = Neuron_Output[2][i][j] * (1-Neuron_Output[2][i][j])
    Node_PD.append(PD)

    # 计算δ
    Delta = []
    Delta.append((Neuron_Output[2]-Target)* Node_PD[0])
    Delta.append(signal.convolve2d(Delta[0], Weight[1], mode = 'same'))

    # 在输入矩阵周围补零
    Add_Zero_Matrix = []
    New_Size = Picture_Size + Convolution_kernel_size - 1
    for i in range(n+1):
        Add_Zero_Matrix.append(np.zeros((New_Size, New_Size), float))
    New_start = int((Convolution_kernel_size - 1) / 2)
    for k in range(n+1):
        for i in range(Picture_Size):
            for j in range(Picture_Size):
                Add_Zero_Matrix[k][New_start+i][New_start+j] = Neuron_Output[k][i][j]

    # 计算权误差
    Error = []
    for i in range(n+1):
        Error.append(signal.convolve2d(Add_Zero_Matrix[i], Delta[i], mode = 'valid'))

    # 修正权
    for i in range(n+1):
        Weight[i] = Weight[i] - η * Error[i]
    
    print(E_total)