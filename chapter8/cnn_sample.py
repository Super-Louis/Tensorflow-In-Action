# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: cnn_sample.py
# Python  : python3.6
# Time    : 18-7-10 23:36
# Github  : https://github.com/Super-Louis

import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

HIDDEN_SIZE = 30  # lstm中隐藏节点个数？隐藏状态的维度？
NUM_LAYERS = 2  # lstm层数

TIMESTEPS = 10  # 循环神经网络训练序列长度
TRAINING_STEPS = 10000  #训练轮数
BATCH_SIZE = 32  # batch大小

TRAINING_EXAMPLES = 10000  # 训练数据个数
TESTING_EXAMPLES = 1000  # 测试数据个数
SAMPLE_GAP = 0.01  # 采样间隔

def generate_data(seq):
    X = []
    y = []
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]]) # 长度为TIMESTEPS的序列
        y.append([seq[i + TIMESTEPS]]) # 序列之后一点的函数值
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def lstm_model(X, y, is_training):
    # 使用多层lstm结构
    cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for
         _ in range(NUM_LAYERS)]
    )
    # 使用tensorflow接口将多层lstm结构连接成RNN网络并计算其前项传播结果
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # outputs是顶层LSTM在每一步(X中每一点)的输出结果
    # 维度为[batch_size, time, HIDDEN_SIZE]？HIDDEN_SIZE
    # 本问题只关注最后一个时刻的输出结果
    output = outputs[:, -1, :]
    # output通过全连接层得到预测值
    predictions = tf.contrib.layers.fully_connected(
        outputs, 1, activation_fn = None
    )
    # 只在训练时计算损失函数和优化步骤
    if not is_training:
        return predictions, None, None

    # 计算损失函数
    # 平均平方差损失函数
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    # 创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer='Adagrad', learning_rate=0.1
    )
    return predictions, loss, train_op