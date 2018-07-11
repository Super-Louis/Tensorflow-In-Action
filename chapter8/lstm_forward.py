# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: lstm_forward.py
# Python  : python3.6
# Time    : 18-7-10 22:01
# Github  : https://github.com/Super-Louis

"""
LSTM前向传播过程
"""
import tensorflow as tf

# 定义lstm结构，LSTM中的变量会自动声明
lstm_hidden_size = ...
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

# 初始化lstm状态，state是一个包含两个张量的LSTMStateTuple类，state.c
# 和state.h分别对应c状态和h状态
batch_size = ...
state = lstm.zero_state(batch_size, tf.float32)

# 定义损失函数
loss = 0.0

def full_connected(output):
    pass

def calc_loss(p, e):
    pass

# num_step为假定的序列长度，实际为一可变长序列
num_steps = ...
for i in range(num_steps):
    # 在第一个时刻申明LSTM结构中使用的变量，之后将复用这些变量
    if i > 0:
        tf.get_variable_scope().reuse_variables()
    # 由当前输入与状态得到当前输出(ht)与更新后的状态(ht, ct)
    current_input = ...
    lstm_output, state = lstm(current_input, state)
    # 将当前时刻LSTM的输出传入全连接层得到最后输出
    final_output = full_connected(lstm_output)
    # 计算当前时刻输出的损失
    expected_output = ...
    loss += calc_loss(final_output, expected_output)



