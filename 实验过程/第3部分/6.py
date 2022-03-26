#!/usr/bin/env python3

# GAN生成器训练部分
import numpy as np
import math

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
# 神经网络节点层输入输出构建
for i in range(n+2):
    Neuron_Input.append(np.zeros((Neuron_Num), float))
    Neuron_Output.append(np.zeros((Neuron_Num), float))
# 神经网络权值矩阵层构建
for i in range(n+1):
    Weight.append(np.random.rand(Neuron_Num,Neuron_Num))
# 输出层误差集合定义
E = np.zeros((Neuron_Num), float)
# 学习率设定
η = 0.5
# 设定训练误差级数
E_level = 2

# 读取数据集
with open('GAN_Target.csv', 'r') as f:
    Original_String = f.read()

First_Process = Original_String.split(',')
Second_Process = list(map(float, First_Process))
# 目标值设定
Target = np.array(Second_Process)
# 底层神经元设定
Neuron_Input[0] = np.random.rand(Neuron_Num)
Neuron_Output[0] = Neuron_Input[0]
E_total = 1
while E_total > 10 ** (-E_level):
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

    # 修正权
    Error = []
    for i in range(n+1):
        a = Neuron_Output[i].reshape([Neuron_Num,1])
        b = Delta[-1-i].reshape([1,Neuron_Num])
        c = np.dot(a,b)
        Error.append(c)

    for i in range(n+1):
        Weight[i] = Weight[i] - η * Error[i]


with open('GAN生成器权值.csv', 'w') as f:
    n = 1
    for ele in Weight:
        if n == 1:
            n = 0
        else:
            f.write('\n')
        for x in range(Neuron_Num):
            for y in range(Neuron_Num):
                f.write(str(ele[x][y]))
                if x != Neuron_Num-1 or y != Neuron_Num-1:
                    f.write(',')