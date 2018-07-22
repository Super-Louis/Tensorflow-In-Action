# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: gen_datasets.py
# Python  : python3.6
# Time    : 18-7-16 23:57

import tensorflow as tf
import numpy as np

TRAIN_DATA = 'data/ptb.train' # 使用单词编号表示的训练数据
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEPS = 35

# 从文件中读取数据，并返回包含单词编号的数组
def read_data(file_path):
    with open(file_path, 'r') as fin:
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]
    return id_list

def make_batches(id_list, batch_size, num_step):
    # 计算总的batch数量， 每个batch包含的单词数量是batch_size * num_step
    num_batches = (len(id_list) - 1) // (batch_size * num_step)

    # 将数据整理成维度为[batch_size, num_batches * num_step]的二维数组
    data = np.array(id_list[:num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches*num_step])
    # 沿着第二个维度将数据切分为num_batches个batch, 存入一个数组
    data_batches = np.split(data, num_batches, axis=1)

    # 重复上述操作，但是每个位置向右移动一位，生成预测标签
    label = np.array(id_list[1: num_batches*batch_size*num_step+1])
    label = np.reshape(label, [batch_size, num_batches*num_step])
    # 分为num_batch个array
    label_batches = np.split(label, num_batches, axis=1)
    return list(zip(data_batches, label_batches))

def main():
    train_batches = make_batches(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEPS)
    ...

if __name__ == '__main__':
    main()