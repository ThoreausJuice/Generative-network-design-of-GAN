#!/usr/bin/env python3

# 单隐层全连接
# 对于本例，使用单个训练集：随机给定输入及初始权值，希望神经网络输出理想图片
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
Neuron_Num = 784
# 神经网络节点层输入输出构建
for i in range(n+2):
    Neuron_Input.append(np.zeros((Neuron_Num), float))
    Neuron_Output.append(np.zeros((Neuron_Num), float))
# 神经网络权值矩阵层构建
for i in range(n+1):
    Weight.append(np.random.rand(Neuron_Num,Neuron_Num))
# 底层神经元设定
Neuron_Input[0] = np.random.rand(Neuron_Num)
Neuron_Output[0] = Neuron_Input[0]
# 目标值设定
# Target = np.array([0,1,0,0,1,0,0,1,0])

# 读取数据集
with open('0样本.csv', 'r') as f:
    Original_String = f.read()

First_Process = Original_String.split('\n')

for ele in First_Process[0:1]:
    Second_Process = ele.split(',')
    Third_Process = list(map(int, Second_Process))
    Forth_Process = list(map(lambda x : x/255, Third_Process))
    Target = np.array(Forth_Process[1:])

# 输出层误差集合定义
E = np.zeros((Neuron_Num), float)
# 学习率设定
η = 0.5
# 设定训练次数
xlcs = 1001

# # 计算开始
for dqcs in range(xlcs):
    # 正向计算
    for j in range(n+1):
        Neuron_Input[j+1] = np.dot(Neuron_Output[j], Weight[j])
        # if j%2 == 0:
        #     Neuron_Input[j+1] += 0.35
        # else:
        #     Neuron_Input[j+1] += 0.60
        for i in range(len(Neuron_Input[j+1])):
            Neuron_Output[j+1][i] = 1 / (1 + math.exp(-Neuron_Input[j+1][i]))
            # if Neuron_Output[j+1][i] < 0:
            #     Neuron_Output[j+1][i] = 0
        # Neuron_Output[j+1] = list(map(lambda: 1 / (1 + math.exp(x)), Neuron_Input[j+1]))

    # 计算总误差
    for i in range(len(E)):
        E[i] = 0.5 * (Target[i] - Neuron_Output[2][i]) ** 2
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
    Delta.append((Neuron_Output[2]-Target)* Node_PD[0])# Node_PD[0]
    for i in range(1,n+1):
        Delta.append(np.dot(Delta[i-1],Weight[len(Weight)-i].T) * Node_PD[0])# Node_PD[i]

    # 修正权
    Error = []
    for i in range(n+1):
        a = Neuron_Output[i].reshape([Neuron_Num,1])
        b = Delta[-1-i].reshape([1,Neuron_Num])
        c = np.dot(a,b)
        Error.append(c)

    for i in range(n+1):
        Weight[i] = Weight[i] - η * Error[i]

    # print('第', int(dqcs), '次训练：')
    # print('误差：', E_total)

    # print(Neuron_Output[0])

    if dqcs%10 == 0:
        print('第', int(dqcs/10), '次训练：')
        # print('输出：', Neuron_Output[2])
        print('误差：', E_total)

# 将每行数据还原成矩阵的函数
def Build_Picture(Picture_List):
    # 构建空矩阵
    # 设定矩阵大小
    n = 28
    Picture_Matrix = np.zeros((n,n), float)
    # 矩阵赋值到哪的光标
    i = 0
    # 将空矩阵进行赋值
    for line in range(n):
        for column in range(n):
            Picture_Matrix[line][column] = Picture_List[i]
            i += 1
    return Picture_Matrix

Input_Picture = Build_Picture(Neuron_Input[0]*255)
Output_Picture = Build_Picture(Neuron_Output[2]*255)
Target_Picture = Build_Picture(Target*255)
from scipy import misc
misc.imsave('Input_Picture.bmp', Input_Picture)
misc.imsave('Output_Picture.bmp', Output_Picture)
misc.imsave('Target_Picture.bmp', Target_Picture)