# -*-coding:utf-8 -*-

'''
@File       : pgn_transformer.py
@Author     : HW Shen
@Date       : 2020/8/20
@Desc       : Encoder, Decoder, PGNTransformer
'''

import tensorflow as tf
from QuestionAnswerSummaryAndReasoning.seq2seq_transformer_pgn_tf2.encoders.self_attention_encoder import EncoderLayer
from QuestionAnswerSummaryAndReasoning.seq2seq_transformer_pgn_tf2.decoders.self_attention_decoder import DecoderLayer
from QuestionAnswerSummaryAndReasoning.seq2seq_transformer_pgn_tf2.layers import positional_encoding, point_wise_forward_network
from QuestionAnswerSummaryAndReasoning.seq2seq_transformer_pgn_tf2.layers import MultiHeadAttention, Embedding, create_padding_mask, create_look_ahead_mask
from QuestionAnswerSummaryAndReasoning.seq2seq_transformer_pgn_tf2.utils.decoding import calc_final_dist


class Encoder(tf.keras.layers.Layer):
    """ Encoder部分，包含多个 EncoderLayer """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        # 论文中Encoder stack包含6个 sub encoder-layer，即num_layers=6
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    """ Decoder部分，包含多个 DecoderLayer """

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        assert d_model % num_heads == 0
        self.depth = self.d_model // self.num_heads
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate=rate)

        self.Wh = tf.keras.layers.Dense(1)
        self.Ws = tf.keras.layers.Dense(1)
        self.Wx = tf.keras.layers.Dense(1)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        attention_weights = {}
        out = self.dropout(x, training=training)

        for i in range(self.num_layers):
            out, block1, block2 = self.dec_layers[i](out, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)

        ### 获取 PGN_transformer 的 context_vectors (结合流程图理解)
        enc_out_shape = tf.shape(enc_output)  # (batch_size, input_seq_len)
        # shape: (batch_size, input_seq_len, num_heads, depth)
        conext = tf.reshape(tensor=enc_output, shape=(enc_out_shape[0], enc_out_shape[1], self.num_heads, self.depth))
        context = tf.transpose(a=conext, perm=[0, 2, 1, 3])  # (batch_size, num_heads, target_seq_len, depth)
        context = tf.expand_dims(input=context, axis=2)  # (batch_size, num_heads, 1, input_seq_len, depth)

        attn = tf.expand_dims(input=block2, axis=-1)  # (batch_size, num_heads, target_seq_len, input_seq_len, 1)
        context = context * attn  # (batch_size, num_heads, target_seq_len, input_seq_len, depth)
        # 通过reduce_sum将维度3的input_seq_len去掉
        context = tf.reduce_sum(input_tensor=conext, axis=3)  # (batch_size, num_heads, target_seq_len, depth)
        context = tf.transpose(a=context, perm=[0, 2, 1, 3])  # (batch_size, target_seq_len, num_heads, depth)
        # shape: (batch_size, target_seq_len, d_model)
        context = tf.reshape(tensor=conext, shape=(tf.shape(context)[0], tf.shape(context)[1], self.d_model))

        # P_gens conputing
        a = self.Wx(x)
        b = self.Ws(out)
        c = self.Wh(context)
        p_gens = tf.sigmoid(self.V(a + b + c))

        return out, attention_weights, p_gens


class PGNTransformer(tf.keras.Model):
    """ PGN-transformer 模型 """

    def __init__(self, params):
        super(PGNTransformer, self).__init__()

        self.params = params
        self.embedding = Embedding(params["vocab_size"],
                                   params["d_model"])
        self.encoder = Encoder(params["num_blocks"],
                               params["d_model"],
                               params["num_heads"],
                               params["dff"],
                               params["vocab_size"],
                               params["dropout_rate"])
        self.decoder = Decoder(params["num_blocks"],
                               params["d_model"],
                               params["num_heads"],
                               params["dff"],
                               params["vocab_size"],
                               params["dropout_rate"])
        # 接一个FC层 + softmax() 输出每个词对应的概率
        self.final_layer = tf.keras.layers.Dense(units=params["vocab_size"])

    def call(self, inp, extended_inp, max_oov_len, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        embed_x = self.embedding(inp)
        embed_dec = self.embedding(tar)
        # shape: (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(embed_x, training, enc_padding_mask)
        # shape: (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights, p_gens = self.decoder(embed_dec,
                                                             enc_output,
                                                             training,
                                                             look_ahead_mask,
                                                             dec_padding_mask)
        # shape: (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)
        final_output = tf.nn.softmax(logits=final_output)
        # shape: (batch_size, num_heads, targ_seq_len, inp_seq_len)
        attn_dists = attention_weights['decoder_layer{}_block2'.format(self.params["num_blocks"])]
        # shape: (batch_size, targ_seq_len, inp_seq_len)
        attn_dists = tf.reduce_sum(attn_dists, axis=1) / self.params["num_heads"]

        # 获取 PGN构架下 词向量的的最终分布结果。calc_final_dist跟PGN网络的一样，没有变化
        final_dists = calc_final_dist(extended_inp,
                                      tf.unstack(final_output, axis=1),
                                      tf.unstack(attn_dists, axis=1),
                                      tf.unstack(p_gens, axis=1),
                                      max_oov_len,
                                      self.params["vocab_size"],
                                      self.params["batch_size"])

        outputs = dict(logits=tf.stack(final_dists, 1), attentions=attn_dists)
        return outputs
