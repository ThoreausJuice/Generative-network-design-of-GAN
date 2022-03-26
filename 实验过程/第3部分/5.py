#!/usr/bin/env python3

# GAN？
import numpy as np
import math
from scipy import misc

# 搭建判别器部分

# 各层神经元输入
Neuron_Input = []
# 各层间权值矩阵
Weight = []
# 各层神经元输出
Neuron_Output = []
# 神经网络隐层层数设定
n = 1
# 神经元个数设定
size = 5
Neuron_Num = size * size
# 神经网络节点层输入输出构建
for i in range(n+2):
    Neuron_Input.append(np.zeros((Neuron_Num), float))
    Neuron_Output.append(np.zeros((Neuron_Num), float))
# 输出层误差集合定义
E = np.zeros((Neuron_Num), float)
# 学习率设定
η = 0.1
# 设定训练误差级数
E_level = 4
# 神经网络权值矩阵层构建
with open('判别器权值.csv', 'r') as f:
    Original_String = f.read()

First_Process = Original_String.split('\n')

from My_Function import Build_Matrix

for ele in First_Process:
    Second_Process = ele.split(',')
    Weight.append(Build_Matrix(Second_Process, Neuron_Num))

# 目标值设定
Target = np.zeros((Neuron_Num), float)
Target[0] = 0 #横
Target[1] = 1
# 底层神经元设定
Neuron_Input[0] = np.random.rand(Neuron_Num)
Input_Picture = Build_Matrix(Neuron_Input[0], size)
misc.imsave('GAN随机输入.bmp', Input_Picture)
# Neuron_Input[0] = np.ones((Neuron_Num), float)
Neuron_Output[0] = Neuron_Input[0]
E_total = 1
Number_of_training = 0
while E_total > 10 ** (-E_level) and Number_of_training < 10**(E_level+1):
    # 正向计算
    for j in range(n+1):
        Neuron_Input[j+1] = np.dot(Neuron_Output[j], Weight[j])
        for i in range(len(Neuron_Input[j+1])):
            Neuron_Output[j+1][i] = 1 / (1 + math.exp(-Neuron_Input[j+1][i]))

    # 计算总误差
    for i in range(len(E)):
        E[i] = 0.5 * (Target[i] - Neuron_Output[n+1][i]) ** 2
    E_total = 0
    for ele in E:
        E_total += ele

    # 反向计算
    # 计算并存储每层节点输出对输入的偏导
    Node_PD = []
    for ele in Neuron_Output[-1:0:-1]:
        Node_PD.append(list(map(lambda x: x*(1-x), ele)))

    # 计算δ
    Delta = []
    Delta.append((Neuron_Output[n+1]-Target)* Node_PD[0])# Node_PD[0]
    for i in range(1,n+1):
        Delta.append(np.dot(Delta[i-1],Weight[len(Weight)-i].T) * Node_PD[i])# Node_PD[i]

    # 修正输入
    Error = np.dot(Delta[-1], Weight[0])
    Neuron_Input[0] = Neuron_Input[0] - η * Error
    # Neuron_Output[0] = Neuron_Input[0]
    # for i in range(len(Neuron_Input[0])):
    #     if Neuron_Input[0][i] < 0:
    #         Neuron_Output[0][i] = 0
    for i in range(len(Neuron_Input[0])):
        Neuron_Output[0][i] = 1 / (1 + 2.718**(-Neuron_Input[0][i]))
    
    # if xlcs%10000 == 0:
    #     print(Neuron_Output[0])
    #     print('第', int(xlcs/10000), '万次训练：')
    #     print('误差：', E_total)
    # xlcs += 1

    Number_of_training += 1
    print('第', Number_of_training, '次误差：', E_total)
print(Neuron_Output[0])
print(Neuron_Output[2])

# write_n = 1
# with open('GAN_Target.csv', 'w') as f:
#     for ele in Neuron_Output[0]:
#         if write_n == 1:
#             write_n = 0
#         else:
#             f.write(',')
#         f.write(str(ele))

Input_Picture = Build_Matrix(Neuron_Output[0], size)
misc.imsave('GAN理想输出.jpg', Input_Picture)