# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: seq2seq_decode.py
# Python  : python3.6
# Time    : 18-7-22 22:14

import tensorflow as tf

CHECKPOINT_PATH = ''

# 模型参数，必须与训练时的参数保持一致
HIDDEN_SIZE = 1024 # lstm隐层规模
NUM_LAYERS = 2 # 隐层层数
SRC_VOCAB_SIZE = 10000 # 源语言词汇表大小
TRG_VOCAB_SIZE = 4000 # 目标语言词汇表大小
SHARE_EMB_AND_SOFTMAX = True

# 词汇表中的<sos>和<eos>的ID，<SOS>第一步的输入，
# <eos>最后一步的输出
SOS_ID = 1
EOS_ID = 2

# 定义类来描述模型
class NMTModel():
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构
        # 与训练时的变量相同，以便恢复模型时能复用参数
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)]
        )
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)]
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

    def inference(self, src_input):
        # 虽然输入只有一个句子，但是dynamic_rnn要求输入为batch形式
        # convert_to_tensor将输入转化为tensor
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        # 使用dynamic_rnn构造编码器
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32
            )
        # 设置解码的最大次数，这是为了避免在极端情况下无限循环
        MAX_DEC_LEN = 100
        with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
            # 使用一个变长的tensor_array存储生成的句子
            init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True,
                                        clear_after_read=False)
            # 填入第一个单词<sos>作为解码器的输入
            init_array = init_array.write(0, SOS_ID)
            # 构建初始的循环状态
            # 包括隐藏状态，生成的array，解码步数
            init_loop_var = (enc_state, init_array, 0)

            # tf.while_loop的循环条件
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID),
                    tf.less(step, MAX_DEC_LEN-1)))

            def loop_body(state, trg_ids, step):
                # 读取最后一步输出的单词，并读取其词向量
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

                # 直接调用dec_cell向前计算一步
                dec_outputs, next_state = self.dec_cell.call(
                    state=state, inputs=trg_emb
                )
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                # 选取logit最大的值作为这一步的输出
                logits = (tf.matmul(output, self.softmax_weight)+self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # 将这一步输出的单词写入循环状态的trg_ids中
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1

            # 执行tf.while_loop, 返回最终状态
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var
            )
            return trg_ids.stack()

def main():
    # 定义训练用的循环神经网络模型
    with tf.variable_scope('nmt_model', reuse=None):
        model = NMTModel()
        # 定义一个测试例子
        test_sentence = [90, 13, 9, 689, 4, 2]
        # 建立解码所需要的计算图
        output_op = model.inference(test_sentence)
        sess = tf.Session()
        saver = tf.train.Saver()
        # 恢复模型
        saver.restore(sess, CHECKPOINT_PATH)
        # 读取翻译结果
        output = sess.run(output_op)
        print(output)
        sess.close()