# -*-coding:utf-8 -*-

'''
@File       : pgn.py
@Author     : HW Shen
@Date       : 2020/8/11
@Desc       :
'''

import tensorflow as tf
from QuestionAnswerSummaryAndReasoning.seq2seq_pgn_tf2.encoders import Encoder
from QuestionAnswerSummaryAndReasoning.seq2seq_pgn_tf2.decoders import Decoder, BahdanauAttentionCoverage, Pointer
from QuestionAnswerSummaryAndReasoning.seq2seq_pgn_tf2.utils import decoding
from QuestionAnswerSummaryAndReasoning.utils import load_word2vec


class PGN(tf.keras.Model):
    """
    Pointer-Generator Network
    """

    def __init__(self, params):
        super(PGN, self).__init__()
        self.embedding_matrix = load_word2vec(params)
        self.params = params
        self.encoder = Encoder(vocab_size=params["vocab_size"],
                               embedding_dim=params["embed_size"],
                               enc_units=params["enc_units"],
                               batch_sz=params["batch_size"],
                               embedding_matrix=self.embedding_matrix)
        self.attention = BahdanauAttentionCoverage(units=params['attn_units'])
        self.decoder = Decoder(vocab_size=params["vocab_size"],
                               embedding_dim=params["embed_size"],
                               dec_units=params["dec_units"],
                               batch_sz=params["batch_size"],
                               embedding_matrix=self.embedding_matrix)

        self.pointer = Pointer()  # PGN 的 pgen系数

    def call_encoder(self, enc_input):
        enc_hidden = self.encoder.initialize_hidden_state()
        # [batch_sz, max_train_x, enc_units], [batch_sz, enc_units]
        enc_output, enc_hidden = self.encoder(input_x=enc_input, hidden=enc_hidden)  # 调用了Encoder的call()方法

        return enc_output, enc_hidden

    def call(self, enc_output, dec_hidden, enc_inp, enc_extended_inp, dec_inp, batch_oov_len,
             enc_padding_mask, use_coverage, prev_coverage):

        predictions, attentions, coverages, p_gens = [], [], [], []

        # 通过调用attention,得到decoder第一步所需的context_vector，coverage等值
        # 调用了 BahdanauAttentionCoverage的 call()方法
        context_vector, attn_dist, coverage_next = self.attention(dec_hidden=dec_hidden,  # (16, 256)
                                                     enc_output=enc_output,  # (16, 200, 256)
                                                     enc_padding_mask=enc_padding_mask,  # (16, 200)
                                                     use_coverage=use_coverage,
                                                     prev_coverage=prev_coverage)

        for t in range(dec_inp.shape[1]):
            ###  Teachering Forcing

            # 调用 Decoder的 call() 方法，获得预测结果 pred，decoder的隐层dec_hidden
            dec_x, pred, dec_hidden = self.decoder(input_x=tf.expand_dims(input=dec_inp[:, t], axis=1),
                                                   dec_hidden=dec_hidden,
                                                   enc_output=enc_output,
                                                   context_vector=context_vector)
            # 再次，调用 BahdanauAttentionCoverage的 call()方法
            context_vector, attn_dist, coverage_next = self.attention(dec_hidden=dec_hidden,
                                                                       enc_output=enc_output,
                                                                       enc_padding_mask=enc_padding_mask, # batcher.py文件中判定
                                                                       use_coverage=use_coverage,
                                                                       prev_coverage=coverage_next)
            # 调用 Pointer 的 call() 方法, 获得 pgen 系数
            p_gen = self.pointer(context_vector=context_vector,
                                 hidden_state=dec_hidden,
                                 dec_inp=tf.squeeze(dec_x, axis=1))

            predictions.append(pred)  # 获得这一步的预测结果
            coverages.append(coverage_next)  # 获得下一步的 coverage
            attentions.append(attn_dist)  # 获得这一步的注意力的分布
            p_gens.append(p_gen)  # 获得这一步的p_gen系数

        # 通过 calc_final_dist（）将 vocab_dists 和 attn_dists 加权获得 final_dists
        # final_dists = pgen * vocab_dists + (1-pgen) * attn_dists
        final_dists = decoding.calc_final_dist(_enc_batch_extend_vocab=enc_extended_inp,
                                               vocab_dists=predictions,  # 预测结果
                                               attn_dists=attentions,
                                               p_gens=p_gens,
                                               batch_oov_len=batch_oov_len,
                                               vocab_size=self.params["vocab_size"],
                                               batch_size=self.params["batch_size"])

        if self.params["mode"] == "train":
            outputs = dict(logits=final_dists,
                           dec_hidden=dec_hidden,
                           attentions=attentions,
                           coverages=coverages,
                           p_gens=p_gens)
        else:  # test mode
            outputs = dict(logits=tf.stack(values=final_dists, axis=1),
                           dec_hidden=dec_hidden,
                           attentions=tf.stack(values=attentions, axis=1),
                           coverages=tf.stack(values=coverages, axis=1),
                           p_gens=tf.stack(values=p_gens, axis=1))

        return outputs