# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: rnn_forward.py
# Python  : python3.6
# Time    : 18-7-10 00:18
# Github  : https://github.com/Super-Louis

"""循环神经网络前向传播"""

import numpy as np

X = [1, 2]
state = [0.0, 0.0] # 初始时刻前一状态为0
# 分开定义不同输入部分的权重
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]]) # 状态权重
w_cell_input = np.asarray([0.5, 0.6]) # 输入权重
b_cell = np.asarray([0.1, -0.1]) # 偏置

# 定义用于输出的全连接层参数
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 按照时间顺序执行循环神经网络的前向传播过程
for i in range(len(X)):
    # 计算循环体中的全连接层神经网络
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation) # 当前时刻输出状态, 下一时刻的输入状态

    # 根据当前时刻状态计算最终输出
    final_output = np.dot(state, w_output) + b_output

    # 输出每个时刻的信息
    print("before activation: {}".format(before_activation))
    print("current state: {}".format(state))
    print("output: {}".format(final_output))
