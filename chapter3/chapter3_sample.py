# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: chapter3_sample.py
# Python  : python3.6
# Time    : 18-6-28 00:17
# Github  : https://github.com/Super-Louis

import tensorflow as tf

from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')

# 定义神经网络的前向传播过程
a = tf.matmul(x, w1)
y = tf.sigmoid(tf.matmul(a, w2))

# 定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) +
                                (1-y_) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))


# 设置自适应学习率
global_step = tf.Variable(0) # 训练的次数, 会自动更新
learning_rate = tf.train.exponential_decay(0.1, global_step, 16, 0.96, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
# train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size, 2)

# 定义规则来给出样本的标签
Y = [[int(x1+x2<1)] for x1,x2 in X] # 是一个多维数组

# 创建一个会话来运行程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    steps = 10000
    for i in range(steps):
        # 每次选举batch_size个样本进行训练
        start = (i*batch_size) % data_size # 可以重复选取
        end = min(start+batch_size, data_size)

        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_:Y})
            print("After %d steps, cross entropy on all data is %g" % (i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))

