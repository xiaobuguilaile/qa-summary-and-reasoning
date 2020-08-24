# -*-coding:utf-8 -*-

'''
@File       : common.py
@Author     : HW Shen
@Date       : 2020/8/19
@Desc       :
'''

import tensorflow as tf


def point_wise_forward_network(d_model, dff):
    """
    FFN（2层，一层是relu）
     - 目的：帮网络引入非线性拟合，增强模型的学习效果"""

    return tf.keras.Sequential([
        # 第一层： 激活函数 relu
        tf.keras.layers.Dense(units=dff, activation='relu'),  # (batch_size, seq_len, dff)
        # 第二层： 默认没有激活函数
        tf.keras.layers.Dense(units=d_model)  # (batch_size, seq_len, d_model)
    ])

