from collections import defaultdict
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def read_data(path_1, path_2, path_3):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        # print(f1)
        for line in f1:
            words = line.split()

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by word count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()  # 除去每个词前后的中的空格
                if not i: continue
                i = i if not lower else item.lower()  # 如果是英文，全部转换成小写
                dic[i] += 1
        # sort
        dic = sorted(dic.items(), key=lambda x: (x[1],x[0]), reverse=True)

        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)
    """
    建立项目的vocab和reverse_vocab，vocab的结构是（词，index）
    """
    vocab = [(w, i) for i, w in enumerate(result)]
    reverse_vocab = vocab[::-1]

    return vocab, reverse_vocab


if __name__ == '__main__':
    lines = read_data('{}/data/train_set.seg_x.txt'.format(BASE_DIR),
                      '{}/data/train_set.seg_y.txt'.format(BASE_DIR),
                      '{}/data/test_set.seg_x.txt'.format(BASE_DIR))
    vocab, reverse_vocab = build_vocab(lines)
    save_word_dict(vocab, '{}/data/vocab.txt'.format(BASE_DIR))


    # key_value = {}
    # key_value[2] = 56
    # key_value[1] = 2
    # key_value[5] = 12
    # key_value[4] = 24
    # key_value[6] = 18
    # key_value[3] = 323
    # print(key_value)
    # key_value = sorted(key_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # print("按值(value)排序:")
    # print(key_value)

    # result = ['张三', '王五','等六']
    # vocab = [(w, i) for i, w in enumerate(result)]
    # reverse_vocab = vocab[::-1]
    #
    # print(vocab)
    # print(reverse_vocab)