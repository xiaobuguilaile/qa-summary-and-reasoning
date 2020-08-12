# -*-coding:utf-8 -*-

'''
@File       : train_eval_test.py
@Author     : HW Shen
@Date       : 2020/7/27
@Desc       :
'''

import tensorflow as tf
from QuestionAnswerSummaryAndReasoning.seq2seq_tf2.models.sequence_to_sequence import SequenceToSequence
from QuestionAnswerSummaryAndReasoning.seq2seq_tf2.batcher import batcher, Vocab
from QuestionAnswerSummaryAndReasoning.seq2seq_tf2.train_helper import train_model
from QuestionAnswerSummaryAndReasoning.seq2seq_tf2.test_helper import beam_decode, greedy_decode
from tqdm import tqdm
from QuestionAnswerSummaryAndReasoning.utils import get_result_filename
import pandas as pd
from rouge import Rouge
import pprint
from loguru import logger


def train(params):
    """ 将参数 mode 修改为 train, 开始训练 """

    assert params['mode'].lower() == "train", "change training mode to train"

    vocab = Vocab(vocab_file=params["vocab_path"], max_size=params['vocab_size'])
    # print("true vocab is ", vocab)
    logger.info("true vocab is {}".format(vocab))

    # print("Creating the batcher ...")
    logger.info("Creating the batcher ...")
    batch_data = batcher(vocab, params)  # 生成一个batch的数据

    # print("Building the model ... ")
    logger.info("Building the model ... ")
    s2s_model = SequenceToSequence(params)

    # print("Creating the checkpoint manager")
    logger.info("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
    ckpt = tf.train.Checkpoint(SequenceToSequence=s2s_model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=checkpoint_dir,
                                              max_to_keep=5)  # the number of checkpoints to keep

    ckpt.restore(ckpt_manager.latest_checkpoint)  # 恢复最后1次的checkpoints结果

    if ckpt_manager.latest_checkpoint:
        # print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        logger.info("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        # print("Initializing from scratch. ")
        logger.info("Initializing from scratch. ")

    logger.info("Starting the training ... ")
    train_model(s2s_model, batch_data, params, ckpt, ckpt_manager)

    s2s_model.fit()  # 开始用数据训练模型...


def test(params):
    """ 将 mode修改为 test, 开始测试 """

    assert params["mode"].lower() == "test", "change training mode to 'test' or 'eval'"
    # assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    # print("Building the model ...")
    logger.info("Building the model ...")
    s2s_model = SequenceToSequence(params)

    # print("Creating the vocab ...")
    logger.info("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    # print("Creating the batcher ...")
    logger.info("Creating the batcher ...")
    batch_data = batcher(vocab, params)

    # print("Creating the checkpoint manager")
    logger.info("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
    ckpt = tf.train.Checkpoint(SequenceToSequence=s2s_model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=checkpoint_dir,
                                              max_to_keep=5)  # the number of checkpoints to keep

    # path = params["model_path"] if params["model_path"] else ckpt_manager.latest_checkpoint
    # path = ckpt_manager.latest_checkpoint
    ckpt.restore(ckpt_manager.latest_checkpoint)
    # print("Model restored")
    logger.info("Model restored")

    # 通过 greedy_search 或 beam_search 预测结果
    # if params['greedy_decode']:
        # params['batch_size'] = 512
    predict_result(s2s_model, params, vocab, params['test_save_dir'])

    # for batch in batch_data:
    #     yield beam_decode(s2s_model, batch, vocab, params)


def predict_result(model, params, vocab, result_save_path):
    """ 通过 greedy_deocde 预测test结果 """
    dataset = batcher(vocab, params)
    # 预测结果
    if params['greedy_decode']:
        results = greedy_decode(model, dataset, vocab, params)  # greedy_search获取结果
    else:
        results = beam_decode(model, dataset, vocab, params)  # beam_search获取结果

    results = list(map(lambda x: x.replace(" ", ""), results))
    # 保存结果
    save_predict_result(results, params)


def save_predict_result(results, params):
    """ 保存test结果到 """
    # 读取结果
    test_df = pd.read_csv(params['test_x_dir'])
    # 填充结果
    test_df['Prediction'] = results[:20000]
    # 提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    result_save_path = get_result_filename(params)
    test_df.to_csv(result_save_path, index=None, sep=',')


def test_and_save(params):
    """ 将测试结果保存到指定位置 """
    # test_save_dir = '{}/data/' (default)
    assert params["test_save_dir"], "provide a dir where to save the results"
    gen = test(params)
    with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
        for i in range(params["num_to_test"]):
            trial = next(gen)
            with open(params["test_save_dir"] + "/article_" + str(i) + ".txt", "w", encoding='utf-8') as f:
                f.write("article:\n")
                f.write(trial.text)
                f.write("\n\nabstract:\n")
                f.write(trial.abstract)
            pbar.update(1)


def evaluate(params):
    """ 评价训练模型结果，以调参优化模型 """
    gen = test(params)
    reals = []
    preds = []
    with tqdm(total=params["max_num_to_eval"], position=0, leave=True) as pbar:
        for i in range(params["max_num_to_eval"]):
            trial = next(gen)
            reals.append(trial.real_abstract)
            preds.append(trial.abstract)
            pbar.update(1)
    r = Rouge()
    scores = r.get_scores(preds, reals, avg=True)
    print("\n\n")
    pprint.pprint(scores)
