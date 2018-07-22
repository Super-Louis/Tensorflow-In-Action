# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: complete_model.py
# Python  : python3.6
# Time    : 18-7-17 22:37

import numpy as np
import tensorflow as tf
from gen_datasets import read_data, make_batches

TRAIN_DATA = 'data/ptb.train' # 训练数据路径
EVAL_DATA = 'data/ptb.valid' # 验证数据路径
TEST_DATA = 'data/ptb.test' # 测试数据路径
HIDDEN_SIZE = 300 # 隐藏层规模

NUM_LAYER = 2 # lstm层数
VOCAB_SIZE = 10000 # 词典规模
TRAIN_BATCH_SIZE = 20 # 训练数据batch大小
TRAIN_NUM_STEP = 35 # 训练数据截断长度

EVAL_BATCH_SIZE = 1 # 测试数据batch的大小
EVAL_NUM_STEP = 1 # 测试数据截断长度
NUM_EPOCH = 5 # 训练轮数
LSTM_KEEP_PROB = 0.9 # LSTM节点不被dropout的概率
EMBEDDING_KEEP_PROB = 0.9 # 词向量不被dropout的概率
MAX_GRAD_NORM = 5 # 用于控制梯度膨胀的梯度大小上限
SHARE_EMB_AND_SOFTMAX = True # 在softmax与词限量层间共享参数


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用的batch大小和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义每一步的输入和预期输出,两者大小均为[batch_size, num_steps]
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义使用lstm作为循环结构且使用dropout的深层循环神经网络
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                output_keep_prob=dropout_keep_prob
            ) for _ in range(NUM_LAYER)
        ]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        # 初始化最初的状态，即全零的向量。
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # 定义单词的词向量矩阵
        embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])

        # 将输入单词转换为词向量
        inputs = tf.nn.embedding_lookup(embedding, self.input_data) # shape: [batch_size, num_steps, HIDDEN_SIZE]

        # 只在训练时使用dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

        # 定义输出列表，在这里先将不同时刻lstm的结构的输出收集起来，再一起提供给softmax层
        outputs = []
        state = self.initial_state
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0: # 复用变量
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state) # input size: [batch_size, hidden_size]
                # cell_output size: [batch_size, hidden_size]
                outputs.append(cell_output)
                # outputs shape: [num_steps, batch_size, hidden_size]
                # 将输出队列展开成[batch_size, hidden_size*num_steps]的形状，
                # 然后再reshape成[batch_size*num_steps, hidden_size]的形状
        # todo: 不懂！！！
        output = tf.reshape(tf.concat(outputs, -1), [-1, HIDDEN_SIZE])

        # softmax层：将rnn在每个位置的输出转化为各个单词的logits
        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable('weight', [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable('bias', [VOCAB_SIZE])
        # logits shape [batch_size*num_steps, vocab_size]
        logits = tf.add(tf.matmul(output, weight), bias, name='logits')

        # 定义交叉熵损失函数和平均损失函数
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]), # 将[batch_size, num_steps] reshape为[batch_size*num_steps]
            logits=logits
        )
        self.cost = tf.reduce_sum(loss) / batch_size # 每个sample在所有time_steps上的损失之和
        self.final_state = state

        # 只有在训练模型时定义方向传播操作
        if not is_training:
            return

        trainable_variables = tf.trainable_variables()
        # 控制梯度大小
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

def run_epoch(session, model, batches, train_op, output_log, step):
    # 计算平均perplexity的辅助变量
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 训练一个epoch
    for x, y in batches: # 每个x, y都是一个batch
        cost, state, _ = session.run(
            [model.cost, model.final_state, train_op],
            {model.input_data: x,
             model.targets: y,
             model.initial_state: state}
        )
        total_costs += cost
        iters += model.num_steps # 所有的time_steps之和

        # 只有在训练时输出日志
        if step % 100 == 0:
            print("after %d steps, perplexity is %.3f" %
                  (step, np.exp(total_costs/iters))) # 每个词的损失
        step += 1
    return step, np.exp(total_costs/iters)

def main():
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练所用的循环神经网络
    with tf.variable_scope('language_model', reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    # 定义测试循环神经网络
    with tf.variable_scope("language_model", reuse=True, initializer=initializer): # 复用language_model中所有变量
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 训练模型
    saver = tf.train.Saver()
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        train_batches = make_batches(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
        eval_batches = make_batches(read_data(EVAL_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP) # 测试截断长度为1
        test_batches = make_batches(read_data(TEST_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)

        step = 0
        for i in range(NUM_EPOCH):
            print("In interation: %d" % (i+1))
            step, train_pplx = run_epoch(session, train_model, train_batches, train_model.train_op,
                                         True, step)
            print("Epoch: %d Train Perplexity: %.3f" % (i+1, train_pplx))
            _, eval_pplx = run_epoch(session, eval_model, eval_batches, tf.no_op(), False, 0)
            print('Epoch: %d Eval Perplexity: %.3f' %(i+1, eval_pplx))
        _, test_pplx = run_epoch(session, eval_model, test_batches, tf.no_op(), False, 0)
        print('Test Perplexity: %.3f' % (test_pplx))
        saver.save(session, 'nlp_model.ckpt')

if __name__ == '__main__':
    main()

