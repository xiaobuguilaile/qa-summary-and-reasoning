# -*-coding:utf-8 -*-

'''
@File       : run_decoder.py
@Author     : HW Shen
@Date       : 2020/8/11
@Desc       :
'''

import tensorflow as tf


class BahdanauAttentionCoverage(tf.keras.layers.Layer):

    def __init__(self, units):
        super(BahdanauAttentionCoverage, self).__init__()
        self.W_h = tf.keras.layers.Dense(units)
        self.W_s = tf.keras.layers.Dense(units)
        self.W_c = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output, enc_padding_mask, use_coverage=False, prev_coverage=None):
        """
        引入coverage机制，计算 v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
        :param dec_hidden: shape=(batch_size, hidden size)=(16, 256)
        :param enc_output: shape=(16, 200, 256)
        :param enc_padding_mask: shape=(16, 200)
        """
        dec_hidden_with_time_axis = tf.expand_dims(input=dec_hidden, axis=1)  # shape==((16, 1, 256))

        def masked_attention(score):
            """
            获取注意力分布 attn_dist
            eg. input : [[3,5,68,2,0,0,0], [12,23,5,0,0,0,0]]
              => mask: [[0,0,0,0,1,1,1], [0,0,0,1,1,1,1]] 类似于取反
            """
            attn_dist = tf.squeeze(input=score, axis=2)  # shape=(16, 200)
            # Calculate Attention Distribution, 归一化score, 得到 attn_dist。
            attn_dist = tf.nn.softmax(logits=attn_dist, axis=1)  # shape==(16, 200, 1)
            mask = tf.cast(x=enc_padding_mask, dtype=attn_dist.dtype)
            attn_dist *= mask
            masked_sums = tf.reduce_sum(input_tensor=attn_dist, axis=1)
            attn_dist = attn_dist / tf.reshape(tensor=masked_sums, shape=[-1, 1])
            attn_dist = tf.expand_dims(input=attn_dist, axis=2)
            return attn_dist

        # coverage机制
        if use_coverage and prev_coverage is not None: # 如果不是第一步的话，prev_coverage不为空
            # 计算 v^T tanh(W_h h_i + W_s s_t + w_c c_i^T + b_attn)
            score = self.V(tf.nn.tanh(self.W_h(enc_output) + self.W_s(dec_hidden_with_time_axis) + self.W_c(prev_coverage)))
            attn_dist = masked_attention(score)  # 获得注意力分布
            coverage = attn_dist + prev_coverage  # 更新coverage参数
        else:
            # 计算 v^T tanh(W_h h_i + W_s s_t + b_attn)
            score = self.V(tf.nn.tanh(self.W_h(enc_output) + self.W_s(dec_hidden_with_time_axis)))
            attn_dist = masked_attention(score)  # 获得注意力分布 # shape=(16, 200, 1)
            # 如果是训练第一步的话，使用coverage得话，给定一个初始值即可
            if use_coverage:
                coverage = attn_dist  # initialize coverage
            else: # 不使用coverage，直接为空 []
                coverage = []

        # context_vector, shape after sum == (batch_size, hidden_size)
        context_vector = attn_dist * enc_output  # shape==(16, 200, 256)
        # tf.reduce_sum()用于计算张量tensor沿着某一维度的和，可以在求和后降维。下面是对第2维求和，实现降维
        context_vector = tf.reduce_sum(input_tensor=context_vector, axis=1)  # shape==(16, 256)
        # tf.squeeze()函数的作用是从tensor中删除所有大小(size)是1的维度。
        attn_dist = tf.squeeze(input=attn_dist, axis=-1)  # shape==(16, 200)

        return context_vector, attn_dist, coverage


class Decoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix):
        """
        vocab_size: 词表的词数（对应params['vocab_size']）, 一般取 30000 or 50000
        embedding_dim: 词向量维度，256
        dec_units: decoder层的单位数
        batch_size: 每批次的样本数
        embedding_matrix: 预训练好的词向量矩阵
        """
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        # 定义Embedding层，加载与训练的词向量
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        # 定义单层的 GRU（or LSTM）
        self.gru = tf.keras.layers.GRU(units=self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # self.dropout = tf.keras.layers.Dropout(0.5)

        # 定义最后的FC层（接softmax()归一化），用于预测词的概率 (这里需要预测每个词的概率，所以单位是vocab_size)
        self.fc = tf.keras.layers.Dense(units=vocab_size,
                                        activation=tf.keras.activations.softmax)

    def call(self, input_x, dec_hidden, enc_output, context_vector):
        """
        input_x: 输入样本
        dec_hidden: 隐层
        enc_output: encoder层的输出
        context_vector: 上下文向量
        """
        # enc_output shape == (batch_size, max_length, hidden_size)
        # context_vector shape == (batch_size, embedding_dim)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(input_x)

        # 先将 context_vector 增加一个维度（第2维），然后跟x在最后1维（第3维）上合并
        x = tf.concat(values=[tf.expand_dims(input=context_vector, axis=1), x], axis=-1)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)

        # passing the concatenated vector x to GRU
        # 注意：因为tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')，
        # from_logits = False表示preds已经经过了softmax
        output, state = self.gru(x)  # 所以这里的 output，是经过 softmax()后的结果
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(tensor=output, shape=(-1, output.shape[2]))
        output = self.fc(output) # 最终的预测结果 preds

        return x, output, state


class Pointer(tf.keras.layers.Layer):
    """
    Pointer-Generator Networks中的 pointer类, 用于定义系数 Pgen 的实现
    """

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_c_reduce = tf.keras.layers.Dense(1)
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, hidden_state, dec_inp):
        # pgen = Sigmoid(w_c_reduce^T * context_vector + w_s_reduce^T * hidden_state + w_i_reduce^T * dec_inp  )
        pgen = tf.nn.sigmoid(self.w_c_reduce(context_vector) + self.w_s_reduce(hidden_state) + self.w_i_reduce(dec_inp))

        return pgen