# -*-coding:utf-8 -*-

'''
@File       : lr_schedules.py
@Author     : HW Shen
@Date       : 2020/8/20
@Desc       :
'''

import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    可变化学习率
      - 学习率先升高，再降低
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        # 学习率升高的步数
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(x=step)  # 平方根倒数计算，即 1/sqrt(step)
        arg2 = step * (step.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
