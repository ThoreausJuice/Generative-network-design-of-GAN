# 神经网络误差反传播二分类器,累积BP版本
# 输入：x, y
# 目标：将 x >= y 的分为[1,0]，x < y 的分为[0,1]

import numpy as np
import math

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
# 学习率设定
η = 0.5
# 累积误差设定
Cumulative_BP = np.array([0, 0], float)


# 读取训练集↓
with open('1.csv', 'r') as f:
    Original_String = f.read()

First_Process = Original_String.split('\n')

for ele in First_Process:
    Second_Process = ele.split(',')
    # 底层神经元设定
    Neuron_Input[0] = np.array([float(Second_Process[0]), float(Second_Process[1])])
    Neuron_Output[0] = Neuron_Input[0]
    # 目标值设定
    Target = np.array([float(Second_Process[2]), float(Second_Process[3])])
    # 计算开始
    # 正向计算
    for j in range(n+1):
        Neuron_Input[j+1] = np.dot(Neuron_Output[j], Weight[j])
        if j%2 == 0:
            Neuron_Input[j+1] += 0.35
        else:
            Neuron_Input[j+1] += 0.60
        for i in range(len(Neuron_Input[j+1])):
            Neuron_Output[j+1][i] = 1 / (1 + math.exp(-Neuron_Input[j+1][i]))

    # # 计算总误差
    # for i in range(len(E)):
    #     E[i] = 0.5 * (Target[i] - Neuron_Output[2][i]) ** 2
    # E_total = 0
    # for ele in E:
    #     E_total += ele

    # 变更累积误差
    Cumulative_BP += Neuron_Output[2]-Target

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
    Weight[i] = Weight[i] - η * Error[i]
