# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: data_processing.py
# Python  : python3.6
# Time    : 18-7-16 23:20

import codecs
import collections
from operator import itemgetter

RAW_DATA = 'data/simple-examples/ptb.valid.txt' # 训练集数据文件
VOCAB_OUTPUT = 'data/ptb.vocab' # 输出的词汇表文件
OUTPUT_DATA = 'data/ptb.valid'

# todo: 标点符号怎么处理
def form_voc():
    counter = collections.Counter()
    with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    # 按照词频顺序对单词进行排序
    sorted_word_to_cnt = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    sorted_words = ['<eos>'] + sorted_words # '<eos>'换行结束符
    # sorted_words = ['<unk>', '<sos>', '<eos>'] + sorted_words

    with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word+'\n')

def encode_sentence():
    with codecs.open(VOCAB_OUTPUT, 'r', 'utf-8') as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]

    # 建立词汇到单词编号的映射
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    # 如果出现被删除的高频词， 则替换为‘《unk》'
    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id['<unk>']

    fin = codecs.open(RAW_DATA, 'r', 'utf-8')
    fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')
    for line in fin:
        words = line.strip().split() + ['<eos>'] # 读取单词并添加结束符，每个句子最后一个单词的预测输出应为结束符
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)
    fin.close()
    fout.close()

if __name__ == '__main__':
    encode_sentence()
