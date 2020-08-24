# -*-coding:utf-8 -*-

'''
@File       : pgn_transformer.py
@Author     : HW Shen
@Date       : 2020/8/19
@Desc       :
'''


import tensorflow as tf
from QuestionAnswerSummaryAndReasoning.seq2seq_transformer_pgn_tf2.layers.position import positional_encoding


def create_padding_mask(seq):
    """ 输入值的 mask，用于padding值softmax()的屏蔽计算 """

    seq = tf.cast(x=tf.math.equal(seq, 0), dtype=tf.float32)
    # 添加额外的维度将填充加到注意力对数（logits)
    return seq[:, tf.newaxis, tf.newaxis, :]  # batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """ decoder里的遮盖机制 mask """
    mask = 1 - tf.linalg.band_part(input=tf.ones((size, size)), num_lower=-1, num_upper=0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    """ 整合: 生成attention所需的各种 mask """

    # Encoder层输入时的 padding mask
    enc_padding_mask = create_padding_mask(inp)

    # decoder层的第二个attention block, 用于 mask encoder层的输出outputs(作为decoder的input)
    dec_padding_mask = create_padding_mask(inp)

    # decoder层的第一个attention block，用于padding和mask decoder层的future tokens
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    # 就是说，这里既要考虑padding的mask,还要考虑前瞻 mask
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def scaled_dot_product_attention(q, k, v, mask):
    """
    计算注意力权重
    - q,k,v必须具有匹配的前置维度
    - k,v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v
    - 虽然 mask 根据其类型（填充或前瞻）有不同的形状，但是 mask 必须能进行广播转换以便求和
    Args
        q: 请求形状 == （..., seq_len_q, depth）
        k: 请求形状 == （..., seq_len_k, depth）
        v: 请求形状 == （..., seq_len_v, depth）
        mask: float张量， 其形状能转换成（..., seq_len_q, seq_len_k）, 默认为None.
    Return: output=scaled_input_x, attention_weights
    """
    # tf.matmul()矩阵a 乘以矩阵b，transpose_b=True表示b计算前转置
    matmul_qk = tf.matmul(a=q, b=k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 对 matmul_qk 进行缩放，从（0，dk）归一化到（0，1）的正态分布
    dk = tf.cast(x=tf.shape(k)[-1], dtype=tf.float32)  # 通过 k 矩阵获取标量维度 dk
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # 除以根号 dk

    # 将mask加入到缩放的张量 scaled_attention_logits
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # 乘以-1e9后，计算softmax()相应值为0

    # softmax()在最后一个轴上（即axis=-1，seq_len_k轴）上归一化，加和为1
    attention_weights = tf.nn.softmax(logits=scaled_attention_logits, axis=-1)

    scaled_input_x = tf.matmul(a=attention_weights, b=v)  # (..., seq_len_q, depth_v)
    return scaled_input_x, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """ 多头注意力 """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0  # 512 / 8 = 64

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads  # 64 = 512 / 8

        # 先作出3个维度为 d_model（eg.512）的大向量(Wq, Wk, Wv)，跟输入x相乘后得到大的（Q,K,V），再对它们切分成num_heads个小的 (q, k, v)
        self.Wq = tf.keras.layers.Dense(d_model)  # d_model * d_model
        self.Wk = tf.keras.layers.Dense(d_model)  # d_model * d_model
        self.Wv = tf.keras.layers.Dense(d_model)  # d_model * d_model

        # 这里是最后要乘的 Wo，d_moel * d_model
        self.dense = tf.keras.layers.Dense(d_model)  # 最后输出时需要的全连接层

    def split_heads(self, x, batch_size):
        """分拆最后一维到 （num_heads, depth）"""

        x = tf.reshape(tensor=x, shape=(batch_size, -1, self.num_heads, self.depth))
        # 转置后形状为：（batch_size, num_heads, seq_len, depth）
        x = tf.transpose(a=x, perm=[0, 2, 1, 3])  # 将 1和 2 维调换
        return x

    def call(self, x, mask):

        batch_size = tf.shape(x)[0]
        Q = self.Wq(x)  # (batch_size, seq_len, d_model)
        K = self.Wk(x)  # (batch_size, seq_len, d_model)
        V = self.Wv(x)  # (batch_size, seq_len, d_model)
        # 注意上下看维度的变化
        # 将 Q最后一维(d_model) 拆分开 放在 q的 1(num_heads) 和 3(depth) 的维度位置上
        q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(K, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        v = self.split_heads(V, batch_size)  # (batch_size, num_heads, seq_len_q, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # shape == (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(a=scaled_attention, perm=[0, 2, 1, 3])
        # 将num_heads和depth维度合并，得到shape == (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(tensor=scaled_attention,
                                      shape=(batch_size, -1, self.d_model))
        # 这里的ouput，就是输出的 Z
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class Embedding(tf.keras.layers.Layer):
    """
    Embedding层
    - 因为输入不再是 w2v 的词向量了，而是需要训练获得的embedding
    """
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size  # embedding 维度  eg.300
        self.d_model = d_model  # 模型维度 eg.512
        # 输入维度vocab_size, 输出维度 d_model
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_encoding = positional_encoding(vocab_size, d_model)

    def call(self, x):
        embed_x = self.embedding(x)  # shape == (batch_size, target_seq_len, d_model)
        embed_x *= tf.math.sqrt(tf.cast(x=self.d_model, dtype=tf.float32))
        embed_x += self.pos_encoding[:, tf.shape(x)[1], :]
        return embed_x



if __name__ == '__main__':
    # 注意： tf.matmul()和tf.multiply()的区别
    a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
    # [[1, 2, 3]
    # [4, 5, 6]]
    b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
    # [[7, 8]
    # [9, 10]
    # [11,12]]
    res = tf.matmul(a, b)
    print(res)

