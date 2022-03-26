#!/usr/bin/env python3

# 卷积运算函数


import numpy as np
from scipy import signal
from My_Function import *

x = np.ones((3,3))
x[0][0] = 1
x[0][1] = 3
x[1][0] = 2
x[1][1] = 2


y = np.ones((3,3))

y[0][0] = 0.1
y[0][1] = 0.2
y[1][0] = 0.2
y[1][1] = 0.4

# z = signal.fftconvolve(x, y, mode = 'full')
y = Flip180(y)
z1 = signal.convolve2d(x, y, mode = 'valid')
# z1 = Flip180(z1)
z2 = signal.convolve2d(z1, x, mode = 'same')
print(z1)