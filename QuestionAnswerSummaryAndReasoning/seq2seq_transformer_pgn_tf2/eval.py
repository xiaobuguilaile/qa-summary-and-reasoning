# -*-coding:utf-8 -*-

'''
@File       : eval.py
@Author     : HW Shen
@Date       : 2020/8/19
@Desc       :
'''

# from tqdm import tqdm
# from rouge import Rouge
# import pprint


# def evaluate(params):
#     gen = test(params)
#     reals = []
#     preds = []
#     with tqdm(total=params["max_num_to_eval"], position=0, leave=True) as pbar:
#         for i in range(params["max_num_to_eval"]):
#             trial = next(gen)
#             reals.append(trial.real_abstract)
#             preds.append(trial.abstract)
#             pbar.update(1)
#     r = Rouge()
#     scores = r.get_scores(preds, reals, avg=True)
#     print("\n\n")
    # pprint.pprint(scores)
