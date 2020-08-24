# -*-coding:utf-8 -*-

'''
@File       : losses.py
@Author     : HW Shen
@Date       : 2020/8/19
@Desc       :
'''

import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')


def loss_function(real, outputs, padding_mask, cov_loss_wt, use_converage):
    """
    根据是否使用coverage机制，选择不同损失函数
    """
    pred = outputs["logits"]
    attn_dists = outputs["attentions"]

    if use_converage:  # 如果使用coverage机制
        # PGN的损失函数 （其中 _coverage_loss 需要的是 α 和 c, c又是α的加和）
        loss = pgn_log_loss_function(real, pred, padding_mask) + cov_loss_wt * _coverage_loss(attn_dists, padding_mask)
        return loss
    else:
        # Seq2seq的损失函数
        return seq2seq_loss_function(real, pred, padding_mask)


def seq2seq_loss_function(real, pred, padding_mask):
    """
    Seq2seq的损失函数
    real: (16，50)
    pred: (16, 50, 30000)
    """
    loss = 0
    for t in range(real.shape[1]):
        loss_ = loss_object(real[:, t], pred[:, t])
        mask = tf.cast(x=padding_mask[:, t], dtype=loss_.dtype)
        mask = tf.cast(x=mask, dtype=loss_.dtype)
        loss_ *= mask
        loss_ = tf.reduce_sum(loss_)
        loss_ += loss_
    return loss / real.shape[1]


def pgn_log_loss_function(real, final_dists, padding_mask):
    """
    计算每一步的 loss
    """
    loss_per_step = []
    batch_nums = tf.range(0, limit=real.shape[0])  # shape (batch_size)
    for dec_step, dist in enumerate(final_dists):
        targets = real[:, dec_step]  # target words
        indices = tf.stack(values=(batch_nums, targets), axis=1)  # tagerts的indices, shape (batch_size, 2)
        # 按照indices的格式从params中抽取切片（合并为一个Tensor）
        gold_probs = tf.gather_nd(params=dist, indices=indices)  # shape (batch_size). prob of correct words on this step
        losses = -tf.math.log(x=gold_probs)
        loss_per_step.append(losses)
    # 加入 dec_padding_mask
    _loss = _mask_and_avg(loss_per_step, padding_mask)

    return _loss


def _mask_and_avg(loss_per_step, padding_mask):
    """
    将 mask 添加到每一步的loss中
    padding_mask： Tensor("Cast_2:0", shape=(64, 400), dtype=float32)
    """
    padding_mask = tf.cast(padding_mask, dtype=loss_per_step[0].dtype)
    dec_lens = tf.reduce_sum(input_tensor=padding_mask, axis=1)
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(loss_per_step)]
    # normalized value for each batch member
    values_per_ex = sum(values_per_step) / dec_lens

    return tf.reduce_mean(values_per_ex)


def _coverage_loss(attn_dists, padding_mask):
    """
    计算 attn_dist 中的 loss
        Args:
      attn_dists: The attention distributions for each decoder timestep.
      A list length max_dec_steps containing shape (batch_size, attn_length)
      padding_mask: shape (batch_size, max_dec_steps).
    Returns:
      coverage_loss: scalar
    """
    # coverage的初始值为0
    coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length).
    # decoder过程中每一步的coverage loss
    covlosses = []
    for a in attn_dists:
        # 计算这一步的 coverage loss
        covloss = tf.reduce_sum(input_tensor=tf.minimum(a, coverage), axis=1)
        covlosses.append(covloss)
        coverage += a
    # 加入 dec_padding_mask
    coverage_loss = _mask_and_avg(covlosses, padding_mask)

    return coverage_loss


if __name__ == '__main__':

    a = tf.constant([[1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15]])
    # index_a1 = tf.constant([[0, 2], [0, 4], [2, 2]])
    # 选出第0列第3个，第0列第5个，第2列第3个
    # print(tf.gather_nd(a, index_a1))  # tf.Tensor([ 3  5 13], shape=(3,), dtype=int32)

    res = tf.reduce_sum(input_tensor=(a), axis=1)
    print(res)  # tf.Tensor([15 40 65], shape=(3,), dtype=int32)

