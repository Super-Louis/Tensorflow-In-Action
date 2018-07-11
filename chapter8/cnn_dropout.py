# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: cnn_dropout.py
# Python  : python3.6
# Time    : 18-7-10 23:17
# Github  : https://github.com/Super-Louis

"""
带dropout的cnn前向传播
"""
import tensorflow as tf
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell

# 通过两个参数：input_keep_prob, output_keep_prob来控制dropout
# 在使用了DropoutWrapper的基础上定义MultiRNNCell
lstm_size = ...
num_of_layers = ...
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
    tf.nn.rnn_cell.DropoutWrapper(lstm_cell(lstm_size))
    for _ in range(num_of_layers)
)
...