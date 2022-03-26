#!/usr/bin/env python3

New_String = []
with open('1样本.csv', 'w') as New_F:
    for i in range(10):
        with open(str(i)+'.csv', 'r') as f:
            Original_String = f.read()

        First_Process = Original_String.split('\n')
        for line in First_Process:
            Second_Process = line.split(',')
            if Second_Process[0] == '1' :#or Second_Process[0] == '1':
                New_F.write(line + '\n')