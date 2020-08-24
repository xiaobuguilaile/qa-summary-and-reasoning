# -*-coding:utf-8 -*-

'''
@File       : self_attention_decoder.py
@Author     : HW Shen
@Date       : 2020/8/19
@Desc       :
'''

import tensorflow as tf
from QuestionAnswerSummaryAndReasoning.seq2seq_transformer_pgn_tf2.layers.transformer import MultiHeadAttention
from QuestionAnswerSummaryAndReasoning.seq2seq_transformer_pgn_tf2.layers.common import point_wise_forward_network


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        # 跟 EncoderLayer 的区别：除了self-attentions之外，Decoder多一个 encoder-decoder attation层
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate=rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=rate)

    def call(self, y, enc_output, training, look_ahead_mask, padding_mask):
        """ Decoder 输入的是 y标签，而 Encoder输入的是x"""

        ### self.attention
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(y, y, y, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        # 残差网络
        out1 = self.layernorm1(attn1 + y)

        ### encoder-decoder attention
        # 输入的 k,v来自 enc_output, 而 q 来自 decoder自身的输出 out1
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        # 残差网络
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        # 经过FFN来提供网络的非线性拟合能力
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2
