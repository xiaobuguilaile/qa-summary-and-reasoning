# -*-coding:utf-8 -*-

'''
@File       : decoding.py
@Author     : HW Shen
@Date       : 2020/8/11
@Desc       :
'''


import tensorflow as tf


def calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
    """
    获取 PGN构架下 词向量的的最终分布结果.
    计算方法： final_dists = p_gen * vocab_dists + (1-p_gen) * attn_dists
    Args:
    vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                The words are in the order they appear in the vocabulary file.
    attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
    Returns:
    final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    # vocab_dists 和 attn_dists 分别乘以相应的系数
    vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
    attn_dists = [(1 - pgen) * dist for (pgen, dist) in zip(p_gens, attn_dists)]

    # 扩增（concat）一些0值到 vocab_dists, 以接收 OOV词的概率结果
    extended_size = vocab_size + batch_oov_len  # 每个batch扩增oov后的vocab_size
    extra_zeros = tf.zeros((batch_size, batch_oov_len))  # 需要填充的0值
    vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

    # 将 atten_dists 的值，投射到 final_dists的相应位置。这意味着如果 a_i=0.1同时对应的word的index是500，
    # 那么encoder输入的第i个词，就将0.1加到final_dists中index=500位置的词的概率上
    # tf.range()创建一个数字序列,并以增量形式(默认间隔 delte=1)扩展，直到但不包括limit
    batch_nums = tf.range(start=0, limit=batch_size)  # shape (batch_size)
    batch_nums = tf.expand_dims(input=batch_nums, axis=1)  # shape (batch_size, 1)

    attend_len = tf.shape(input=_enc_batch_extend_vocab)[1]  # 已参与的states数量
    # tf.tile() 平铺，用于在同一维度上的复制。如下，将batch_nums在
    batch_nums = tf.tile(input=batch_nums, multiples=[1, attend_len])   # shape (batch_size, attend_len)
    indices = tf.stack(values=(batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
    shape = [batch_size, extended_size]
    attn_dists_projected = [tf.scatter_nd(indices=indices, updates=copy_dist, shape=shape) for copy_dist in attn_dists]

    # 将 vocab_dists_extended 和 attn_dists_projected 对应相加得到最终结果
    final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

    return final_dists


if __name__ == '__main__':

    # nums = tf.constant([1,2])
    # res = tf.tile(nums, [1,3])
    # print(res)  # tf.Tensor([1 2 1 2 1 2], shape=(6,), dtype=int32)

    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    shape = tf.constant([8])
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)  # tf.Tensor([ 0 11  0 10  9  0  0 12], shape=(8,), dtype=int32)