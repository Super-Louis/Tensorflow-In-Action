# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: study_mnist_intro.py
# Python  : python3.6
# Time    : 18-6-24 22:52
# Github  : https://github.com/Super-Louis

import tensorflow as tf
import input_data

# 导入数据集
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

# 创建参数及变量
x = tf.placeholder('float', [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 定义模型
y = tf.nn.softmax(tf.matmul(x, w)+b)
y_ = tf.placeholder('float', [None, 10])

# 定义损失函数及训练算法
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 变量及模型初始化
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 模型训练重复1000次，每批选取100条数据进行批量更新
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) # 小批次学习
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # 输入x, y_会自动计算y, cross_entropy

# 模型评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # 返回每个输出向量的最大值所对应的索引，即预测label与实际label
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float')) # cast a tensor to a new type
print(sess.run(accuracy, feed_dict={x: mnist.test.images[:1000], y_: mnist.test.labels[:1000]}))

