# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: mnist_eval.py
# Python  : python3.6
# Time    : 18-7-2 00:32
# Github  : https://github.com/Super-Louis

# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: mnist.eval.py
# Python  : python3.6
# Time    : 18-7-2 22:15
# Github  : https://github.com/Super-Louis

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference, mnist_train

# 每10s加载一次模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        # 因为测试时不关注正则化损失的值，所以正则化损失函数为None
        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型
        variables_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY
        )
        variables_to_restore = variables_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 每隔一段时间计算一次准确率
        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH
                )
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型, 加载模型时，所有变量需要重新定义
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时的迭代轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("after %d training step(s), validation accuracy is %g."
                          % (int(global_step), accuracy_score))
                else:
                    print('no checkpoint found, return')
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()

