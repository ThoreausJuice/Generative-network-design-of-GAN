#!/usr/bin/env python3

# 可变隐层数全连接判别器，使用训练好的权值将横竖图片分为两类
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
size = 5
Neuron_Num = size * size
# 神经网络节点层输入输出构建
for i in range(n+2):
    Neuron_Input.append(np.zeros((Neuron_Num), float))
    Neuron_Output.append(np.zeros((Neuron_Num), float))
# 输出层误差集合定义
E = np.zeros((Neuron_Num), float)
# 神经网络权值矩阵层构建
with open('判别器权值.csv', 'r') as f:
    Original_String = f.read()

First_Process = Original_String.split('\n')

from My_Function import Build_Matrix

for ele in First_Process:
    Second_Process = ele.split(',')
    Weight.append(Build_Matrix(Second_Process, Neuron_Num))

# 读取数据集
with open('横'+str(size)+'×'+str(size)+'.csv', 'r') as f:
    Original_String = f.read()

with open('竖'+str(size)+'×'+str(size)+'.csv', 'r') as f:
    # Original_String = f.read()
    Original_String += '\n'+f.read()

First_Process = Original_String.split('\n')

# 正确个数
correct_num = 0
# 测试个数
test_num = 0
for ele in First_Process:# 训练集范围更改在这里
    test_num += 1
    Second_Process = ele.split(',')
    Third_Process = list(map(int, Second_Process))
    # 目标值设定
    Target = np.zeros((Neuron_Num), float)
    Lable = int(Third_Process[0])
    if Lable == 0:
        Target[0] = 1
    else:
        Target[1] = 1
    print(Target)
    # 底层神经元设定
    Neuron_Input[0] = np.array(Third_Process[1:])
    Neuron_Output[0] = Neuron_Input[0]
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
    
    Probability_Sum = 0
    for ele in Neuron_Output[-1][0:2]:
        Probability_Sum += ele
    Probability_0 = Neuron_Output[-1][0] / Probability_Sum * 100
    Probability_1 = Neuron_Output[-1][1] / Probability_Sum * 100

    if (Probability_0 > Probability_1 and Target[0] == 1) or (Probability_0 < Probability_1 and Target[0] == 0):
        print('横：', Probability_0, '%  竖：', Probability_1, '%')
        print('正确')
        correct_num += 1
    else:
        print('横：', Probability_0, '%  竖：', Probability_1, '%')
        print('错误')


print('正确率：', correct_num/test_num*100, '%')