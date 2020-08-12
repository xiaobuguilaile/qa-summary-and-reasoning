# -*-coding:utf-8 -*-

'''
@File       : eval.py
@Author     : HW Shen
@Date       : 2020/8/11
@Desc       :
'''

from QuestionAnswerSummaryAndReasoning.seq2seq_pgn_tf2.train_eval_test import test
from tqdm import tqdm
from rouge import Rouge
import pprint


def eveluate(params):
    """ 评估 """

    gen = test(params)
    reals = []
    preds = []
    with tqdm(total=params["max_num_to_eval"], position=0, leave=True) as pbar:
        for i in range(params["max_num_to_eval"]):
            trail = next(gen)
            reals.append(trail.real_abstract)
            preds.append(trail.abstract)
            pbar.update(1)
    r = Rouge()
    scores = r.get_scores(preds, reals, avg=True)
    print("\n\n")
    pprint.pprint(scores)

