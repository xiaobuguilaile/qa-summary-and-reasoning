import numpy as np
import pandas as pd
import re
from jieba import posseg
import jieba
from QuestionAnswerSummaryAndReasoning.utils.tokenizer import segment
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


REMOVE_WORDS = ['|', '[', ']', '语音', '图片', ' ']


def read_stopwords(path):
    """移除停止词"""
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


def remove_words(words_list):
    """移除干扰词"""
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list


def preprocess_sentence(sentence):
    """句子清洗"""
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_list = remove_words(seg_list)
    seg_line = ' '.join(seg_list)
    return seg_line


def parse_data(train_path, test_path):
    """数据预处理，并保存在指定文件"""
    train_df = pd.read_csv(train_path, encoding='utf-8')
    train_df.dropna(subset=['Report'], how='any', inplace=True)  # 移除report为NAN的记录
    train_df.fillna('', inplace=True)  # 将其余的NAN用 空字符''表示
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    print('train_x is ', len(train_x))
    train_x = train_x.apply(preprocess_sentence)
    print('train_x is ', len(train_x))
    train_y = train_df.Report
    print('train_y is ', len(train_y))
    train_y = train_y.apply(preprocess_sentence)
    print('train_y is ', len(train_y))
    # if 'Report' in train_df.columns:
        # train_y = train_df.Report
        # print('train_y is ', len(train_y))

    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_x = test_x.apply(preprocess_sentence)
    print('test_x is ', len(test_x))
    test_y = []
    train_x.to_csv('{}/data/train_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)
    train_y.to_csv('{}/data/train_set.seg_y.txt'.format(BASE_DIR), index=None, header=False)
    test_x.to_csv('{}/data/test_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)


if __name__ == '__main__':
    # 需要更换成自己数据的存储地址
    parse_data('{}/data/AutoMaster_TrainSet.csv'.format(BASE_DIR),
               '{}/data/AutoMaster_TestSet.csv'.format(BASE_DIR))

    # print(BASE_DIR)

