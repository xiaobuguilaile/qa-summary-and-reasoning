# -*-coding:utf-8 -*-

'''
@File       : test_helper.py
@Author     : HW Shen
@Date       : 2020/8/2
@Desc       :
'''

import tensorflow as tf
import numpy as np
from QuestionAnswerSummaryAndReasoning.seq2seq_tf2.batcher import output_to_words
from tqdm import tqdm


def greedy_decode(model, dataset, vocab, params):
    """
    采用 greedy search 完成 decode
    model: seq2seq model
    dataset: 测试数据集
    vocab：词表
    params: 参数集
    """
    batch_size = params['batch_size']

    # 存储结果
    results = []
    sample_size = 20000  # 测试集样本数量
    # batch 操作轮数，match.ceil()向上取整。因为最后一个batch可能不足batch_size,但不能丢掉
    step_epoches = sample_size // batch_size + 1

    for i in tqdm(range(step_epoches)):
        enc_data, _ = next(iter(dataset))  # 迭代器不断返回 enc_batch_data
        results += batch_greedy_decode(model, enc_data, vocab, params)

    return results


def batch_greedy_decode(model, enc_data, vocab, params):

    # 判断输入长度
    batch_data = enc_data["enc_input"]
    batch_size = enc_data["enc_input"].shape[0]
    # 预测结果存储list
    predicts = [''] * batch_size
    # inputs == batch_data  # shape=(3,115)
    inputs = tf.convert_to_tensor(batch_data)  # 将batch_data转换为张量
    # enc_hidden = [tf.zeros((batch_size, params['enc_units']))
    # enc_output, enc_hidden = model.encoder(inputs,enc_hidden)
    enc_output, enc_hidden = model.call_encoder(inputs)

    # dec_input = tf.expand_dims([vocab.word_to_id(vocab.START_DECODING)] * batch_size, 1)
    dec_inputs = tf.constant([2] * batch_size)  # [2,2,2,2...]
    dec_inputs = tf.expand_dims(input=dec_inputs, axis=1)  # [[2],[2],[2],[2]...]

    dec_hidden = enc_hidden  # 将encoder的最后一个隐层作为decoder的输入隐层
    context_vector, _ = model.attention(dec_hidden=dec_hidden,
                                        enc_output=enc_output)
    for t in range(params['max_dec_len']):  # 遇到 [END] or params['max_dec_len'] 结束
        # step predict
        x, pred, dec_hidden = model.decoder(input_x=dec_inputs,
                                            dec_hidden=dec_hidden,
                                            enc_output=enc_output,
                                            context_vector=context_vector)
        context_vector, _ = model.attention(dec_hidden=dec_hidden,
                                            enc_output=enc_output)
        # 最后需要numpy()将tensor张量转化为ndarray数组
        predicted_ids = tf.argmax(input=pred, axis=1).numpy()
        for index, predicted_id in enumerate(predicted_ids):
            predicts[index] += vocab.id_to_word(predicted_id)  # 将序号组成的句子，转化回word构成的句子

        # using teacther forcing
        dec_inputs = tf.expand_dims(predicted_ids, 1)

    results = []
    for predict in predicts:
        predict = predict.strip()  # 去掉句子前后的空格
        # 如果句子长度小于max_len就结束了，那么说明句子中存在'[STOP]'，句子的结果就到vocab.word_to_id('[STOP]')为止。
        if '[STOP]' in predict:
            predict = predict[:predict.index('[STOP]')]
        # 保存结果
        results.append(predict)
    return results


########## beam search part  ################


class Hypothesis(object):
    """
    Class designed to hold hypothesises throughout the beam_search decoding
    到时刻T，有N种可能（假设），每种假设计算从time=0到time=T时刻, 所有tokens的概率的对数和。
    后面会根据概率大小选出beam_size个最大的 "可能"
    """
    def __init__(self, tokens, log_probs, state, attn_dists, prob_gens):

        self.tokens = tokens  # list of all the tokens from Time_0 to the Current_time_step t
        self.log_probs = log_probs  # list of the log_probabilities of the tokens 所有tokens的对数和
        self.state = state  # decoder state after the last token decoding
        self.attn_dists = attn_dists  # attention dists of all the tokens
        self.prob_gens = prob_gens  # generation probability of all the tokens

    def extend(self, token, log_prob, state, attn_dist, prob_gen):
        """ Method to extend the current hypothesis by adding next decoded token and all information associated """
        # 将新产生的decoded结果加到相应的列表中
        return Hypothesis(tokens=self.tokens + [token],  # add decoded token
                          log_probs=self.log_probs + [log_prob],  # add log_prob of the decoded token
                          state=state,  # update the state, 状态值不需要累加，更新传递即可
                          attn_dists=self.attn_dists + [attn_dist],  # add_attn dist of the decoded token
                          prob_gens=self.prob_gens + [prob_gen])  # add prob_gen of the decoded token

    @property
    def latest_token(self):
        return self.tokens[-1]  # 返回最后一个 token

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)  # 返回所有log和

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)  # 返回log的平均值


def beam_decode(model, batch, vocab, params):
    """ 采用 beam search 完成 decode """

    def decode_onestep(enc_input, enc_outputs, dec_input, dec_state, enc_extended_inp,
                       batch_oov_len, enc_pad_mask, use_coverage, prev_coverage):
        """
        单步 decoder, 目的是为了得到一步 decode 的 top_k 的中间结果
        Method to decode the output step by step (used for beamSearch decoding)
        Args:
            batch : current batch, shape = [beam_size, 1, vocab_size( + max_oov_len if pointer_gen)] (for the beam search decoding, batch_size = beam_size)
            enc_outputs : hiddens outputs computed by the encoder BiGRU
            dec_input : decoder_input, the previous decoded batch_size-many words, shape = [beam_size, embed_size]
            dec_state : beam_size-many list of decoder previous state, GRUStateTuple objects, shape = [beam_size, 2, hidden_size]
            prev_coverage : beam_size-many list of previous coverage vector
        Returns: A dictionary of the results of all the ops computations (see below for more details)
        """
        # seq2seq模型, beam_size=3, embedding_size=256
        preds, dec_hidden, attentions, prob_gens = model(enc_outputs,  # shape=(3, 115, 256)
                                                               dec_state,  # shape=(3, 256)
                                                               enc_input,  # shape=(3, 115)
                                                               enc_extended_inp,  # shape=(3, 115)
                                                               dec_input,  # shape=(3, 1)
                                                               batch_oov_len,  # shape=()
                                                               enc_pad_mask,  # shape=(3, 115)
                                                               use_coverage,
                                                               prev_coverage)  # shape=(3, 115, 1)

        top_k_probs, top_k_ids = tf.nn.top_k(input=tf.squeeze(preds), k=params["beam_size"] * 2)
        # 对 top_k_probs 取 ln(以e为底), eg. [0.4, 0.3,0.1] => [-0.9162907 -1.2039728 -2.3025851]
        top_k_log_probs = tf.math.log(x=top_k_probs)
        # 返回需要保存的中间结果和概率
        results = {"dec_state": dec_hidden,
                   "attention_vec": attentions,
                   "top_k_ids": top_k_ids,
                   "top_k_log_probs": top_k_log_probs,
                   "prob_gens": prob_gens}

        return results

    # 计算 encoder的输出，作为decode的输入：token_state shape=(3, 256), enc_outputs shape=(3, 115, 256)
    enc_outputs, state = model.call_encoder(enc_input=batch[0]["enc_input"])

    # 初始化 beam_size个 Hypothesis对象，长度为 beam_size (beam_size-many list)
    hyps = [Hypothesis(tokens=[vocab.word_to_id('[START]')],
                       log_probs=[0.0],
                       state=[0],
                       prob_gens=[],
                       attn_dists=[])
            for _ in range(params['beam_size'])]

    results = []  # # 初始化结果集，就是最终结果 (the top beam_size hypothesis)
    steps = 0  # 遍历的步数
    while steps < params["max_dec_steps"] and len(results) < params["beam_size"]:
        # 获取最新待使用的 token，在第一步的时候就是[START]，单步的 hyps 一开始是 1
        latest_tokens = [h.latest_token for h in hyps]
        # 用 UNKNOWN token替换所有的 OOV
        # latest_tokens = [t if t in vocab.id2word else vocab.word2id('[UNK]') for t in latest_tokens]
        latest_tokens = [t if t in range(params["voacb_size"]) else vocab.word_to_id('[UNK]') for t in latest_tokens]
        # 获取每个 hypothesis 的隐藏层状态
        states = [h.state for h in hyps]
        dec_states = tf.stack(values=states, axis=0)  # 在 axis=0 增加一个维度
        # 最新输入decode的 dec_input 就是最后面的 latest_tokens. We decode the top-likely 2*beam_size tokens at step t for each hypothesis
        dec_input = tf.expand_dims(input=latest_tokens, axis=1)  # shape: (3,0) => (3, 1)

        returns = decode_onestep(enc_input=batch[0]['enc_input'],  # shape=(3, 115)
                                 enc_outputs=enc_outputs,  # shape=(3, 115, 256)
                                 dec_input=dec_input,  # shape=(3, 1)
                                 dec_state=dec_states,  # shape=(3, 256)
                                 enc_extended_inp=batch[0]['extended_enc_input'],
                                 batch_oov_len=batch[0]['max_oov_len'],
                                 enc_pad_mask=batch[0]['sample_encoder_pad_mask'],
                                 use_coverage=params['is_coverage'],
                                 prev_coverage=None)

        topk_ids, topk_log_probs, new_states, attn_dists, prob_gens = \
            returns['top_k_ids'], returns['top_k_log_probs'], returns['dec_state'],  returns['attention_vec'], returns["prob_gens"]

        all_hyps = []  # 当前全部可能情况
        num_orig_hyps = 1 if steps == 0 else len(hyps)  # 原有的可能情况数量

        # 遍历添加所有可能结果，第一步的时候 num_orig_hyps 就是1
        for i in range(num_orig_hyps):
            # h, new_state, attn_dist, p_gen, coverage = hyps[i], new_states[i], attn_dists[i], p_gens[i], prev_coverages[i]
            h, new_state, attn_dist, prob_gen = hyps[i], new_states[i], attn_dists[i], prob_gens[i]
            for j in range(params["beam_size"] * 2):
                # we extend each hypothesis with each of the TOP_K tokens
                # (this gives 2*beam_size new hypothesises for each of the beam_size old hypothesises)
                new_hyp = h.extend(token=topk_ids[i, j],
                                   log_prob=topk_log_probs[i, j],
                                   state=new_state,
                                   attn_dist=attn_dist,
                                   prob_gen=prob_gen)
                all_hyps.append(new_hyp)

        # Then, we sort all the hypothesises, and select only the beam_size most likely hypothesises
        hyps = []
        sorted_hyps = sorted(all_hyps, key=lambda h:h.avg_log_prob, reverse=True)  # 排序用的是平均值 avg_log_prob
        # 通过一次遍历，把不符合长度要求的 h 剔除
        for h in sorted_hyps:
            if h.latest_token == vocab.word_to_id('[STOP]'):
                # 长度如果符合预期的话, 遇到句尾,添加到结果集
                if steps >= params["min_dec_steps"]:  # decoder需要steps下限
                    results.append(h)
            else:
                hyps.append(h)  # 未到结尾, 把h添加到假设集

            # 如果假设句子正好等于 beam_size 或者 结果集正好等于 beam_size 就不再添加
            if len(hyps) == params["beam_size"] or len(results) == params["beam_size"]:
                break
        steps += 1

    if len(results) == 0:  results = hyps  # 如果没有合适结果，就把原 hyps 当作 results

    # 循环结束后，对 most-likely 的 hyps进行排序，选出最可能的 best_hyps
    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    best_hyp = hyps_sorted[0]
    # 摘要
    best_hyp.abstract = " ".join(output_to_words(best_hyp.tokens, vocab, batch[0]["article_oovs"][0])[1:-1])
    # 全文
    best_hyp.text = batch[0]["article"].numpy()[0].decode()

    return best_hyp


if __name__ == '__main__':
    # inputs = tf.constant([2] * 5)  # ?
    # print(inputs.shape)
    # inputs = tf.expand_dims(input=inputs, axis=1)
    # print(inputs.shape)

    # pred = tf.constant([[1, 2, 3],
    #                      [6, 2, 4],
    #                      [123, 17, 1]])
    # res = tf.argmax(input=pred, axis=0).numpy()
    # print(res)

    # top_k_probs = [0.4, 0.3,0.1]
    # top_k_log_probs = tf.math.log(x=top_k_probs)
    # print(top_k_log_probs)
    # tf.Tensor([-0.9162907 -1.2039728 -2.3025851], shape=(3,), dtype=float32)

    states1 = tf.constant([1, 2, 3, 4])  # shape=(4,)
    states2 = tf.constant([1, 2, 3, 4])  # shape=(4,)
    res = tf.stack(values=[states1, states2], axis=0)  # shape=(2, 4)
    print(res)
