# -*-coding:utf-8 -*-

'''
@File       : position.py
@Author     : HW Shen
@Date       : 2020/8/19
@Desc       : transformer的位置参数
'''

import tensorflow as tf
import numpy as np


def get_angels(pos, i, d_model):
    """ 获取sin(), cos()的内部角度 """
    angel_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))

    return pos * angel_rates


def positional_encoding(position, d_model):
    """
    获取位置参数
    position：
    d_model:
    """
    # np.newaxis的功能:插入新维度
    angel_rads = get_angels(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # 偶数位置，将sin应用于数组中的偶数索引（indices）: 2i
    angel_rads[:, 0::2] = np.sin(angel_rads[:, 0::2])
    # 奇数位置，将cos用于数组中的奇数索引：2i+1
    angel_rads[:, 1::2] = np.cos(angel_rads[:, 1::2])

    pos_encoding = angel_rads[np.newaxis, ...]

    return tf.cast(x=pos_encoding, dtypr=tf.float32)


if __name__ == '__main__':
    n = np.arange(12).reshape(3, 4)
    print(n)
    # [[ 0  1  2  3]
    #  [ 4  5  6  7]
    #  [ 8  9 10 11]]
    print(n[:, 0::2])
    # [[ 0  2]
    #  [ 4  6]
    #  [ 8 10]]
    print(n[:, 1::2])