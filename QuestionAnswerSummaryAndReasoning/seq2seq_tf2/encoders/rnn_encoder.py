# -*-coding:utf-8 -*-

'''
@File       : rnn_encoder.py
@Author     : HW Shen
@Date       : 2020/7/27
@Desc       :
'''

import tensorflow as tf


class Encoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz  # 每批次输入多少个样本（句子）
        self.enc_units = enc_units // 2  # ? 为什么除以2
