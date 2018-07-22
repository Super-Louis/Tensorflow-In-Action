# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: seq2seq_test.py
# Python  : python3.6
# Time    : 18-7-18 23:47

import tensorflow as tf

MAX_LEN = 50 # 限定句子的最大单词数量
SOS_ID = 1 # 目标语言词汇表中<sos>的id

# 使用dataset从一个文件中读取一个语言的数据
# 数据的格式为每行一句话，单词已转化为单词编号
def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    # 根据空格将单词编号切分开并放入一个一维向量
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 将字符串形式的单词编号转化为整数
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个单词的数量，并与句子内容一起放入Dataset中
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset

# 读取数据，填充及batching
def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)

    # 通过zip操作将两个dataset合二为一， 每个dataset中的每一项数据ds由4个张量组成
    # ds[0][0] 源句子
    # ds[0][1] 源句子长度
    # ds[1][0] 目标句子
    # ds[1][1] 目标句子长度
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # 删除内容为空和过长的句子
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        # 查看logical用法
        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)
    # 保留满足条件的data
    dataset = dataset.filter(FilterLength)
    # 上述目标句子形式为 x, y, z, <eos>，从中生成<sos>, x, y, z形式加入到Dataset中
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, stc_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, stc_len), (trg_input, trg_label, trg_len))
    dataset = dataset.map(MakeTrgInput)

    # 随机打乱数据集
    dataset = dataset.shuffle(10000)

    # 规定填充后输出的数据维度
    padded_shapes = (
        (tf.TensorShape([None]), # 源句子是长度未知的向量
         tf.TensorShape([])), # 源句子长度是单个数字
        (tf.TensorShape(None), # 目标句子(输入)是长度未知的向量
         tf.TensorShape([None]), # 目标句子(输出)是长度未知的向量
         tf.TensorShape([])) # 目标长度为单个数字
    )

    # 调用padded_batch方法进行batching操作
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset