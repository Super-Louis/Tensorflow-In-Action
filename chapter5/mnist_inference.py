# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: mnist_inference.py
# Python  : python3.6
# Time    : 18-7-1 23:15
# Github  : https://github.com/Super-Louis

import tensorflow as tf
# 定义神经网络结构相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 通过tf.get_variable函数来获取变量。在训练神经网络时会创建这些变量；
# 在测试时会通过保存的模型加载这些变量的取值
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 正则化
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

# 定义神经网络的前向传播过程
def inference(input_tensor, regularizer):
    # 不同命名空间的权重定义
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE],
                             initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)+biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE],
                             initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights)+biases
    return layer2