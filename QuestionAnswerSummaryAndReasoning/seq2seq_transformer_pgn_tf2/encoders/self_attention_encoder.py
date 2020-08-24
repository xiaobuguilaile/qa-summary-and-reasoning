# -*-coding:utf-8 -*-

'''
@File       : self_attention_encoder.py
@Author     : HW Shen
@Date       : 2020/8/19
@Desc       :
'''

import tensorflow as tf
from QuestionAnswerSummaryAndReasoning.seq2seq_transformer_pgn_tf2.layers.transformer import MultiHeadAttention
from QuestionAnswerSummaryAndReasoning.seq2seq_transformer_pgn_tf2.layers.common import point_wise_forward_network


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)  # training表示：dropout只是在train时使用，test时不用
        # 残差结构的实现（or 跳跃链接）
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # 残差结构的实现（or 跳跃链接）
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
