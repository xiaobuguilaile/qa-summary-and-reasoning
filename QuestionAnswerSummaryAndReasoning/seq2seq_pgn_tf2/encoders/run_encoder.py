# -*-coding:utf-8 -*-

'''
@File       : run_encoder.py
@Author     : HW Shen
@Date       : 2020/8/11
@Desc       :
'''


import tensorflow as tf


class Encoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        """
        vocab_size: 选取词表的词数，即所有词库单词个数
        embedding_size: 输入样本句子的维度（vocab_size, embedding_dim）
        enc_units: encoder层的单位数 (GRU or LSTM的单位数)
        batch_sz: 每批次输入样本数
        embedding_matrix: embedding的权重矩阵
        """
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz  # 每批次输入多少个样本（句子）
        # self.enc_units = enc_units # 单向
        self.enc_units = enc_units // 2  # 双向GRU，每个GRU均分enc_units
        # 定义Embedding层，加载与训练的词向量
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        # 定义单向的GRU（或LSTM），tf.keras.layers.GRU自动匹配CPU或GPU
        self.gru = tf.keras.layers.GRU(units=self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')  # 参数初始化：基于方差的初始化，适合激活函数为tanh()
        # 定义双向 GRU（或LSTM）
        self.bigru = tf.keras.layers.Bidirectional(layer=self.gru, merge_mode='concat')  # merge_mode 拼接方式为直接拼接

    def call(self, input_x, hidden):
        """
        x: 输入样本
        hidden: 隐藏层
        """
        x = self.embedding(input_x)  # 将输入的样本数据转化为向量
        # 将hidden张量切分成2个子张量
        hidden = tf.split(value=hidden,  # 被切分的对象
                          num_or_size_splits=2,
                          axis=1)
        # output, state = self.gru(x, initial_state=hidden) # 单向GRU
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)  # 列方向相加

        return output, state

    def initialize_hidden_state(self):

        return tf.zeros((self.batch_sz, 2*self.enc_units))

