# 神经网络误差反传播二分类器,标准BP版本
# 输入：x, y
# 目标：将 x >= y 的分为[1,0]，x < y 的分为[0,1]

import numpy as np
import math
from random import *

# 单隐层神经网络搭建↓
# 各层神经元输入
Neuron_Input = []
# 各层间权值矩阵
Weight = []
# 各层神经元输出
Neuron_Output = []
# 神经网络隐层层数设定
n = 1
# 神经网络节点层输入输出构建
for i in range(n+2):
    Neuron_Input.append(np.array([0, 0], float))
    Neuron_Output.append(np.array([0, 0], float))
# 神经网络权值矩阵层构建
# for i in range(n+1):
#     Weight.append(np.random.rand(2,2))
Weight.append(np.array([[0.15, 0.25], [0.20, 0.30]]))
Weight.append(np.array([[0.40, 0.50], [0.45, 0.55]]))
# 输出层误差集合定义
E = np.array([0, 0], float)
# 总训练集输出误差设定
Cumulative_Error = 1
# 误差目标级数设定（影响训练次数）
ET_Num = -2

# 读取训练集↓
with open('1.csv', 'r') as f:
    Original_String = f.read()

First_Process = Original_String.split('\n')

while Cumulative_Error > 10**ET_Num:
    Cumulative_Error = 0
    for ele in First_Process:
        Second_Process = ele.split(',')
        # 底层神经元设定
        Neuron_Input[0] = np.array([float(Second_Process[0]), float(Second_Process[1])])
        Neuron_Output[0] = Neuron_Input[0]
        # 目标值设定
        Target = np.array([float(Second_Process[2]), float(Second_Process[3])])
        # 计算开始
        E_total = 1
        while E_total > 10**(ET_Num-1):
            # 正向计算
            for j in range(n+1):
                Neuron_Input[j+1] = np.dot(Neuron_Output[j], Weight[j])
                if j%2 == 0:
                    Neuron_Input[j+1] += 0.35
                else:
                    Neuron_Input[j+1] += 0.60
                for i in range(len(Neuron_Input[j+1])):
                    Neuron_Output[j+1][i] = 1 / (1 + math.exp(-Neuron_Input[j+1][i]))

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
            Delta.append((Neuron_Output[2]-Target)*Node_PD[0])
            for i in range(1,n+1):
                Delta.append(np.dot(Delta[i-1],Weight[len(Weight)-i].T) * Node_PD[i])

            # 修正权
            Error = []
            for i in range(n+1):
                a = Neuron_Output[i].reshape([2,1])
                b = Delta[-1-i].reshape([1,2])
                c = np.dot(a,b)
                Error.append(c)

            for i in range(n+1):
                # 学习率设定
                η = random()
                Weight[i] = Weight[i] - η * Error[i]
        Cumulative_Error += E_total
    print('训练集总误差：', Cumulative_Error)


print('\n', len(First_Process), '个训练样本训练完毕，使用测试集进行测试↓')
# 测试集测试
# 读取测试集↓
with open('2.csv', 'r') as f:
    Text_Original_String = f.read()

Text_First_Process = Text_Original_String.split('\n')

# 正确答案计数
Correct = 0

# 测试开始
for ele in Text_First_Process:
    Second_Process = ele.split(',')
    # 底层神经元设定
    Neuron_Input[0] = np.array([float(Second_Process[0]), float(Second_Process[1])])
    Neuron_Output[0] = Neuron_Input[0]
    # 目标值设定
    Target = np.array([int(Second_Process[2]), int(Second_Process[3])])
    # 计算开始
    for j in range(n+1):
        Neuron_Input[j+1] = np.dot(Neuron_Output[j], Weight[j])
        if j%2 == 0:
            Neuron_Input[j+1] += 0.35
        else:
            Neuron_Input[j+1] += 0.60
        for i in range(len(Neuron_Input[j+1])):
            Neuron_Output[j+1][i] = 1 / (1 + math.exp(-Neuron_Input[j+1][i]))
    # print(Neuron_Output[2], Target)
    x = int(Neuron_Output[2][0]+0.5)
    y = int(Neuron_Output[2][1]+0.5)
    # print(x,y)
    if x == Target[0] and y == Target[1]:
        Correct += 1

print('总数：', len(Text_First_Process), '正确个数', Correct)
print('正确率：', Correct/len(Text_First_Process)*100, '%')