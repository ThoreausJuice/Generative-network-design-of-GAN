#!/usr/bin/env python3

# 可变隐层数全连接生成器，使用训练好的权值将随机噪音生成横或竖图片
import numpy as np
import math
from scipy import misc

# 各层神经元输入
Neuron_Input = []
# 各层间权值矩阵
Weight = []
# 各层神经元输出
Neuron_Output = []
# 神经网络隐层层数设定
n = 1
# 神经元个数设定
size = 3
Neuron_Num = size * size
# 生成图片个数
Picture_num = 1
# 神经网络节点层输入输出构建
for i in range(n+2):
    Neuron_Input.append(np.zeros((Neuron_Num), float))
    Neuron_Output.append(np.zeros((Neuron_Num), float))
# 神经网络权值矩阵层构建
with open('GAN生成器权值.csv', 'r') as f:
    Original_String = f.read()

First_Process = Original_String.split('\n')

from My_Function import Build_Matrix

for ele in First_Process:
    Second_Process = ele.split(',')
    Weight.append(Build_Matrix(Second_Process, Neuron_Num))

for PN in range(Picture_num):
    # 底层神经元设定
    Neuron_Input[0] = np.random.rand(Neuron_Num)
    Neuron_Output[0] = Neuron_Input[0]
    for j in range(n+1):
        Neuron_Input[j+1] = np.dot(Neuron_Output[j], Weight[j])
        for i in range(len(Neuron_Input[j+1])):
            Neuron_Output[j+1][i] = 1 / (1 + math.exp(-Neuron_Input[j+1][i]))

    Input_Picture = Build_Matrix(Neuron_Input[0]*255, size)
    Output_Picture = Build_Matrix(Neuron_Output[n+1]*255, size)

    misc.imsave('GAN_Input_Picture'+ str(PN) +'.bmp', Input_Picture)
    misc.imsave('GAN_Output_Picture'+ str(PN) +'.bmp', Output_Picture)