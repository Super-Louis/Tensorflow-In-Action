# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: cnn_sample.py
# Python  : python3.6
# Time    : 18-7-10 23:36
# Github  : https://github.com/Super-Louis

import numpy as np
import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot as plt

HIDDEN_SIZE = 30  # lstm中隐藏节点个数, 隐藏状态的维度
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
    output = outputs[:, -1, :] # [batch_size, HIDDEN_SIZE]
    # output通过全连接层得到预测值
    predictions = tf.contrib.layers.fully_connected(
        output, 1, activation_fn = None
    ) # num_outputs --- numbers of outputs,[batch_size,num_outputs]
    # 只在训练时计算损失函数和优化步骤
    if not is_training:
        return predictions, None, None # 输出刚好在sin(x)的范围内

    # 计算损失函数
    # 平均平方差损失函数
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    # 创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer='Adagrad', learning_rate=0.1
    )
    return predictions, loss, train_op

def train(sess, train_X, train_y):
    # 将训练数据以数据集的方式提供给计算图
    # 数据来源于自己生成的array
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()

    # 调用预测模型，得到预测结果、损失函数、和训练操作
    with tf.variable_scope('model'):
        prediction, loss, train_op = lstm_model(X, y, True)

    # 初始化变量
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print("train step: " + str(i) + ", loss: " + str(l))

def run_eval(sess, test_X, test_y):
    # 将测试数据以数据集的方式提供给计算图
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1) # 每次输入一个sample
    X, y = ds.make_one_shot_iterator().get_next()

    # 调用模型得到计算结果，这里不需要输入真实的y值
    with tf.variable_scope('model', reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)

    # 将预测结果存入一个数组
    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    # 计算rmse作为评价指标
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions-labels)**2).mean(axis=0))
    print("Mean square Error is %f" % rmse)

    # 对预测的sin函数曲线进行绘制
    plt.figure()
    plt.plot(predictions, label='predicions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()

# 用正弦函数生成训练和测试数据集合
test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES+TIMESTEPS, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES+TIMESTEPS, dtype=np.float32)))

with tf.Session() as sess:
    train(sess, train_X, train_y) # 将sess传入
    run_eval(sess, test_X, test_y)
