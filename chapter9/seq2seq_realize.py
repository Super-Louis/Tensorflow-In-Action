# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: seq2seq_realize.py
# Python  : python3.6
# Time    : 18-7-19 21:54

import tensorflow as tf
from seq2seq_test import MakeSrcTrgDataset

# 假设输入数据已转换成了单词编号的格式
SRC_TRAIN_DATA = '/path/to/data/train.en' # 源语言输入文件
TRG_TRAIN_DATA = '/path/to/data/train.zh' # 目标语言输入文件
CHECKPOINT_PATH = '/path/to/seq2seq_ckpt' # checkpoint保存路径
HIDDEN_SIZE = 1024 # LSTM隐层规模
NUM_LAYER = 2 # LSTM 层数
SRC_VOCAB_SIZE = 10000 # 源语言词汇表大小
TRG_VOCAB_SIZE = 4000 # 目标语言词汇表大小??
BATCH_SIZE = 100 # 训练数据大小
NUM_EPOCH = 5 # 训练数据轮数
KEEP_PROB = 0.8 # 节点不被dropout的概率
MAX_GRAD_NORM = 5 # 用于控制梯度膨胀的梯度大小上限
SHARE_EMB_AND_SOFTMAX = True # 在softmax层和词向量层间共享参数

# 定义NMTModel类来描述模型
class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYER)]
        )
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYER)]
        )

        # 为源语言和目标语言分别定义词向量
        self.src_embedding = tf.get_variable(
            'src_emb', [SRC_VOCAB_SIZE, HIDDEN_SIZE]
        )
        self.trg_embedding = tf.get_variable(
            'trg_emb', [TRG_VOCAB_SIZE, HIDDEN_SIZE]
        )

        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
                'weight', [HIDDEN_SIZE, TRG_VOCAB_SIZE]
            )
        self.softmax_bias = tf.get_variable(
            'softmax_bias', [TRG_VOCAB_SIZE]
        )

    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        """
        前向传播
        :param src_input: shape [batch_size, padded_len]
        :param src_size: shape [1], true size
        :param trg_input: shape [batch_size, padded_len]
        :param trg_label: shape [batch_size, padded_len]
        :param trg_size: shape [1], true size
        :return:
        """
        batch_size = tf.shape(src_input[0])

        # 将输入和输出单词编号转换为词向量
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # 在词向量上进行dropout
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB) # shape [batch_size, padded_len, hidden_size]
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB) # shape [batch_size, padded_len, hidden_size]

        # 使用dynamic_rnn构造编码器
        # 编码器读取源句子每个位置的词向量，输出最后一步的隐藏状态enc_state
        # enc_state为最后一步的隐藏状态，是一个包含两个LSTMStateTuple类
        # 的tuple, 每个对应一层状态
        # enc_outputs是顶层LSTM在每一步的输出，其维度为[batch_size, padded_len, hidden_size]
        with tf.variable_scope('encoder'):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32
            )

        # 使用dynamic_rnn构造解码器
        # dec_outputs是顶层LSTM在每一步的输出，其维度为[batch_size, padded_len, hidden_size]
        with tf.variable_scope("decoder"):
            dec_outputs, _ = tf.nn.dynamic_rnn(
                self.dec_cell, trg_emb, trg_size, initial_state=enc_state
            )

        # 计算解码器每一步的log perplexity
        # out_put shape: [batch_size*padded_len, hidden_size]
        out_put = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(out_put, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label, [-1]), logits=logits
        ) # labels=tf.reshape(trg_label, [-1]), [batch_size, padded_len] --> [batch_size*padded_len]

        # 在计算平均损失时，需要将填充位置的权重设为0
        # 模型的训练
        label_weights = tf.sequence_mask(
            trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32
        ) # 不需要乘以batch_size?
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss*label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights) # 总损失/token个数

        # 定义反向传播操作
        trainable_variables = tf.trainable_variables()
        # 控制梯度大小，定义优化方法和训练步骤
        grads = tf.gradients(cost/tf.to_float(batch_size), trainable_variables) # 为什么要除以batch_size
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return cost_per_token, train_op

# 使用给定的模型上训练一个epoch, 并返回全局步数
# 每训练200步保存一个checkpoint
def run_epoch(session, cost_op, train_op, saver, step):
    # 训练一个epoch
    # 重复训练直至遍历完所有数据
    while True:
        try:
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print("after %d steps, per token cost is %.3f" % (step, cost))
            # 每200步保存一个checkpoint
            if step % 200 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step

def main():
    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    # 定义训练用的循环神经网络
    with tf.variable_scope("nmt_model", reuse=None, initializer=initializer):
        train_model = NMTModel()
    # 定义输入数据
    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    # 定义前向计算图
    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)

    # 训练模型
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i+1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)

if __name__ == '__main__':
    main()

