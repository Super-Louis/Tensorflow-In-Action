# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: deep_rnn.py
# Python  : python3.6
# Time    : 18-7-10 22:51
# Github  : https://github.com/Super-Louis

"""
深层cnn前向传播
"""
import tensorflow as tf
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell

# 通过MultiRNNCell类实现深层循环网络中的前向传播
lstm_size = ... # what?
num_of_layers = ... # 层数
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
    [lstm_cell(lstm_size) for _ in range(num_of_layers)]
)

# 通过zero_state函数来获取初始状态
batch_size = ...
state = stacked_lstm.zero_state(batch_size, tf.float32) # state大小与层数大小相同？？

def full_connected(output):
    pass

def calc_loss(p, e):
    pass

loss = 0.0
num_steps = ...
for i in range(len(num_steps)):
    if i > 0:
        tf.get_variable_scope().reuse_variables()
    current_input = ...
    stacked_lstm_output, state = stacked_lstm(current_input, state)
    final_output = full_connected(stacked_lstm_output)
    expected_output = ...
    loss += calc_loss(final_output, expected_output)