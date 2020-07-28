# -*-coding:utf-8 -*-

'''
@File       : sequence_to_sequence.py
@Author     : HW Shen
@Date       : 2020/7/27
@Desc       :
'''

import tensorflow as tf
from QuestionAnswerSummaryAndReasoning.seq2seq_tf2.encoders import Encoder
from QuestionAnswerSummaryAndReasoning.seq2seq_tf2.decoders import Decoder, BahdanauAttention
from QuestionAnswerSummaryAndReasoning.utils.data_utils import load_word2vec
import time


class SequenceToSequence(tf.keras.Model):

    def __init__(self, params):
        super(SequenceToSequence, self).__init__()
        self.embedding_matrix = load_word2vec(params)  # 预训练好的词向量权重矩阵
        self.params = params
        self.encoder = Encoder(vocab_size=params["vocab_size"],
                               embedding_dim=params["embed_size"],
                               enc_units=params["enc_units"],
                               batch_sz=params["batch_size"],
                               embedding_matrix=self.embedding_matrix)
        self.attention = BahdanauAttention(units=params['attn_units'])
        self.decoder = Decoder(vocab_size=params["vocab_size"],
                               embedding_dim=params["embed_size"],
                               dec_units=params["dec_units"],
                               batch_sz=params["batch_size"],
                               embedding_matrix=self.embedding_matrix)

    def call_encoder(self, enc_input):
        enc_hidden = self.encoder.initialize_hidden_state()
        # [batch_sz, max_train_x, enc_units], [batch_sz, enc_units]
        enc_output, enc_hidden = self.encoder(input_x=enc_input, hidden=enc_hidden)  # 调用了Encoder的call()方法

        return enc_output, enc_hidden

    def call(self, enc_output, dec_input, dec_hidden, dec_target):
        predictions = []
        attentions = []
        # 调用了 BahdanauAttention的 call()方法
        context_vector, _ = self.attention(dec_hidden=dec_hidden,  # shape=(16, 256)
                                           enc_output=enc_output)  # shape=(16, 200, 256)
        # Teachering Forcing
        for t in range(dec_target.shape[1]):  # 50
            _, pred, dec_hidden = self.decoder(input_x=tf.expand_dims(dec_input[:, t], 1),
                                               dec_hidden=dec_hidden,
                                               enc_output=enc_output,
                                               context_vector=context_vector)
            context_vector, attn_dist = self.attention(dec_hidden=dec_hidden, enc_output=enc_output)

            predictions.append(pred)
            attentions.append(attn_dist)

        # tf.stack()是增加一个维度来拼接，而 tf.concat()只是单纯的拼接
        return tf.stack(predictions, 1), dec_hidden

