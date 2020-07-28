# -*-coding:utf-8 -*-

'''
@File       : train_helper.py
@Author     : HW Shen
@Date       : 2020/7/27
@Desc       :
'''

import tensorflow as tf
import time

STRAT_DECODING = '[START]'


def train_model(seq2seq_model, dataset, params, ckpt, ckpt_manager):
    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=params['learning_rate'],
    #                                         initial_accumulator_value=params['adagrad_init_acc'],
    #                                         clipnorm=params['max_grad_norm'])
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params['learning_rate'])
    # from_logits = True: preds is model output before passing it into softmax (so we pass it into softmax)
    # from_logits = False: preds is model output after passing it into softmax (so we skip this step)

    # 要看模型在decoder最后输出是否经过softmax
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    # 定义损失函数
    def loss_function(real, pred):
        # 逻辑取非 True和False对调 eg.[[[True, Flase...],[]]
        mask = tf.math.logical_not(tf.math.equal(real, 1))
        dec_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=1)  # 沿 axix=1求和

        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss_ = tf.reduce_sum(input_tensor=loss_, axis=1) / dec_lens

        # axis不填，默认所有数取平均
        loss_ = tf.reduce_mean(input_tensor=loss_)
        return loss_

    # @tf.function()
    def train_step(enc_inp, dec_tar, dec_inp):
        # loss=0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = seq2seq_model.call_encoder(enc_inp)
            dec_hidden = enc_hidden
            # start index
            pred, _ = seq2seq_model(enc_output=enc_output,  # shape=(3, 200, 256)
                                    dec_input=dec_inp,  # shape=(3, 256)
                                    dec_hidden=dec_hidden,  # shape=(3, 200)
                                    dec_target=dec_tar)  # shape=(3, 50)
            loss = loss_function(dec_tar, pred)

        # variables = model.trainable_variables
        variables = seq2seq_model.trainable_variables + seq2seq_model.attention.trainable_variables + seq2seq_model.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return loss

    best_loss = 20
    epochs = params['epochs']
    for epoch in range(epochs):
        t0 = time.time()
        step = 0
        total_loss = 0
        # for step, batch in enumerate(dataset.take(params['steps_per_epoch'])):
        # for batch in dataset.take(params['steps_per_epoch']):
        for batch in dataset:
            loss = train_step(enc_inp=batch[0]["enc_input"],  # shape=(16, 200)
                              dec_tar=batch[1]["dec_target"],  # shape=(16, 50)
                              dec_inp=batch[1]["dec_input"])

            step += 1
            total_loss += loss
            if step % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1, step, total_loss/step))
                # print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, loss.numpy()))

        if epoch % 1 == 0:
            if total_loss / step < best_loss:
                best_loss = total_loss / step
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))


if __name__ == '__main__':
    # real = 0
    # print(tf.math.equal(real, 1))
    pass