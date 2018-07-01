# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: mnist_recognizer.py
# Python  : python3.6
# Time    : 18-6-30 21:39
# Github  : https://github.com/Super-Louis

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist数据集相关常数
INPUT_NODE = 784
OUTPUT_NODE = 10

# 配置神经网络的参数
LAYER1_NODE = 500

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8 # 基础的学习率
LEARNING_RATE_DECAY = 0.99 # 学习率的衰减率
REGULARIZATION_RATE = 0.0001 # 正则化项系数
TRAINING_STEPS = 30000 # 训练轮数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率

# 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average函数来计算得出变量的滑动平均值
        # 然后再计算相应的神经网络向前传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))
                            + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 不使用滑动平均
    y = inference(x, None, weights1, biases1, weights2, biases2)


    # 定义存储训练轮数的变量， 由于该变量不使用滑动平均， 因此该变量因为不可训练的参数
    global_step = tf.Variable(0, trainable=False)
    # 使用滑动平均
    variables_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    # 在所有代表神经网络参数的变量上使用滑动平均
    variables_averages_op = variables_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果
    average_y = inference(x, variables_averages, weights1, biases1, weights2, biases2)
    # 损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY
    )
    train_steps = tf.train.GradientDescentOptimizer(learning_rate).\
        minimize(loss, global_step=global_step) # 传入学习率，用于更新学习率
    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络参数，又要更新每个参数的滑动平均值，
    # 为一次完成多个操作，tensorflow提供了tf.control_dependencies和tf.group两种机制
    # 下面的程序和
    # train_op = tf.group(train_step, variables_averages_op)是等价的
    with tf.control_dependencies([train_steps, variables_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("after %d training steps, validation accuracy using average model is %g" %
                      (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("after %d training steps, test accuracy using average model is %g" %
              (TRAINING_STEPS, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run() # tf.app.run()会自动调用main函数