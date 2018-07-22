# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: nlp_theory.py
# Python  : python3.6
# Time    : 18-7-15 16:40
# Github  : https://github.com/Super-Louis
import tensorflow as tf

# 假设词汇表的大小为3，语料包含两个词“2 0”
word_labels = tf.constant([2, 0])

# 假设模型对两个单词预测时，产生的logit分别是[2.0, -1.0, 3.0]和[1.0,0.0,-0.5]
# 注意这里的logit不是概率，如果要计算，则需要调用prob=tf.nn.softmax(logits)
# 但这里计算交叉熵的函数直接输入logits即可
predict_logit = tf.constant([[2.0, -1.0, 3.0], [1.0, 0.0, -0.5]])

# 使用sparse_softmax_cross_entropy_with_logits计算交叉熵
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=word_labels, logits=predict_logit
)

# 运行程序，计算loss的结果是[0.32656264, 0.46436879], 这对应两个预测的perplexity损失
sess = tf.Session()
sess.run(loss)

# softmax_cross_entropy_with_logits与上面的函数类似，但需要将预测目标以概率分布的形式给出
word_prob_distribution = tf.constant([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
loss = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_distribution, logits=predict_logit)
sess.run(loss)

# 由于sotfmax_cross_entropy_with_logits允许提供一个概率分布，因此在使用时有更大的自由度；
# 采用label_smoothing提高训练效果
word_prob_smooth = tf.constant([[0.01, 0.01, 0.98], [0.98, 0.01, 0.01]])
loss = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_distribution, logits=predict_logit)

# 运行结果为[0.37656256, 0.48936883]
sess.run(loss)