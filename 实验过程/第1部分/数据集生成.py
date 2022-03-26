# 数据集生成器
from random import *
biaozhi = 0
with open('1.csv', 'w') as f:
    for i in range(1000):
        x = random()
        y = random()
        if biaozhi == 1:
            f.write('\n')
        if x <= y:
            f.write(str(x)+','+str(y)+',0,1')
        else:
            f.write(str(x)+','+str(y)+',1,0')
        biaozhi = 1
        
biaozhi = 0
with open('2.csv', 'w') as f:
    for i in range(100):
        x = random()
        y = random()
        if biaozhi == 1:
            f.write('\n')
        if x <= y:
            f.write(str(x)+','+str(y)+',0,1')
        else:
            f.write(str(x)+','+str(y)+',1,0')
        biaozhi = 1