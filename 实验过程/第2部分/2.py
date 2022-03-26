#!/usr/bin/env python3
# 以下是我大胆的想法，变卷积核神经网络
# 灵感来自：[Hornik et al., 1989]证明，只需一个包含足够多神经元的隐层，多层前馈网络就能以任意精度逼近任意复杂度的连续函数
import numpy as np
from My_Function import *
from scipy import signal
import math

# 判别器神经网络搭建↓
# 各层神经元输入
Neuron_Input = []
# 各层间权值矩阵
Weight = []
# 各层神经元输出
Neuron_Output = []
# 神经网络隐层层数设定
n = 1
# 神经网络权值矩阵层构建（初始大小预设3×3）
default = 3
Side_Length = 28-default+1
Weight.append(np.random.rand(default, default))
Weight.append(np.random.rand(Side_Length, Side_Length))
# LeakyReLU 斜率
α = 0.2

# 读取数据集
with open('0&1.csv', 'r') as f:
    Original_String = f.read()

First_Process = Original_String.split('\n')

for ele in First_Process[0:1]:
    # 底层神经元输出及标签设定
    Picture_Matrix, Lable = Build_Picture(ele)
    Neuron_Output.append(Picture_Matrix)

    # 计算开始
    # 正向计算
    # 卷积运算
    Feature_Map = signal.convolve(Neuron_Output[0], Weight[0], mode = 'valid')
    Neuron_Input.append(Feature_Map)
    # 激活函数使用LeakyReLU
    LR = np.zeros((Side_Length,Side_Length))
    for x in range(Side_Length):
        for y in range(Side_Length):
            if Neuron_Input[0][x][y] >= 0:
                LR[x][y] = Neuron_Input[0][x][y]
            else:
                LR[x][y] = Neuron_Input[0][x][y] * α
    Neuron_Output.append(LR)
    # 输出层对标签之间的映射使用等大小卷积，取代全连接层(暂定)
    Output_Matrix = Neuron_Output[1]*Weight[1]
    Output = 0
    for x in Output_Matrix:
        for y in x:
            Output += y

    # 使用Sigmoid激活函数
    Output = 1 / (1 + math.exp(-Output))
    Neuron_Output.append(Output)

    # # 使用LeakyReLU激活函数
    # if Output < 0:
    #     Output *= α

    # 计算误差
    E = 0.5 * (Lable - Output) ** 2
    
    # 反向计算
    # 计算并存储每层节点输出对输入的偏导
    Node_PD = []
    # 先载入最后一层
    Node_PD.append(Output*(1-Output))
    # 载入倒数第二层
    PD = np.ones((Side_Length, Side_Length))
    for x in range(Side_Length):
        for y in range(Side_Length):
            if Neuron_Output[1][x][y] < 0:
                PD[x][y] *= α
    Node_PD.append(PD)
        
    # 计算δ
    Delta = []
    Delta.append((Neuron_Output[2]-Lable)*Node_PD[0])
    for i in range(1,n+1):
        Delta.append(np.dot(Delta[i-1],Flip180(Weight[len(Weight)-i])) * Node_PD[i])
    print(Delta)