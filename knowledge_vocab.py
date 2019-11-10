import csv
import numpy as np
import re
import tensorflow as tf

EOS = "[EOS]"


# 除去所有符号
def discrete_string(s1):
    # 把句子按字分开，中文按字分，英文按单词，数字按空格
    res = re.compile(r"([\u4e00-\u9fa5\W])")  # [\u4e00-\u9fa5]中文范围
    str1_list = res.split(s1)
    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符
    return list_word1


def expand_vocab(vocab, dlc):
    discrete = discrete_string(dlc)
    vocab |= set(discrete)
    return discrete


# 注意token的id是从1开始，因为0留个pad
def create_map_from_vocab(f):
    vocab_map = dict()
    for (i, line) in enumerate(f, start=1):
        vocab_map[line.strip()] = str(i)
    return vocab_map


def tokenize_vocab():
    input_vocab = set()
    # 输出时要考虑到EOS (End Of Sequence/)
    output_vocab = set([EOS])
    with open("data/SAOKE_DATA_discrete.csv") as s:
        with open("data/SAOKE_DATA_tokenize.csv", "w") as t:
            reader = csv.reader(s, delimiter="\t")
            writer = csv.writer(t, delimiter="\t")
            writer.writerow(next(reader))
            next(reader)
            for line in reader:
                out = expand_vocab(output_vocab, line[0])
                out.append(EOS)
                writer.writerow([' '.join(expand_vocab(input_vocab, line[0])),
                                 ' '.join(out)])
    with open("data/input_vocab", 'w') as v:
        for (i, w) in enumerate(input_vocab, start=1):
            v.write(w + "\n")
    with open("data/output_vocab", 'w') as v:
        for (i, w) in enumerate(output_vocab, start=1):
            v.write(w + "\n")


def tokenize_id():
    with open("data/SAOKE_DATA_tokenize.csv") as s:
        input_vocab_map = create_map_from_vocab(open("data/input_vocab"))
        output_vocab_map = create_map_from_vocab(open("data/output_vocab"))
        with open("data/SAOKE_DATA_tokenize_id.csv", "w") as t:
            reader = csv.reader(s, delimiter="\t")
            writer = csv.writer(t, delimiter="\t")
            writer.writerow(next(reader))
            for line in reader:
                writer.writerow([
                    ' '.join(map(lambda x: input_vocab_map[x], line[0].split(' '))),
                    ' '.join(map(lambda x: output_vocab_map[x], line[1].split(' '))),
                ])


def string_to_int_array(string, split=' '):
    return list(map(int, string.split(split)))


def save_as_npy_by_train_test(train_weight=9, test_weight=1):
    knowledge_train = []
    natural_train = []
    knowledge_test = []
    natural_test = []
    with open("data/SAOKE_DATA_tokenize_id.csv") as s:
        reader = csv.reader(s, delimiter="\t")
        next(reader)
        for (i, line) in enumerate(reader):
            b = i % (train_weight + test_weight)
            if b < train_weight:
                knowledge_train.append(string_to_int_array(line[0]))
                natural_train.append(string_to_int_array(line[1]))
            else:
                knowledge_test.append(string_to_int_array(line[0]))
                natural_test.append(string_to_int_array(line[1]))
    knowledge_train = np.array(knowledge_train)
    natural_train = np.array(natural_train)
    knowledge_test = np.array(knowledge_test)
    natural_test = np.array(natural_test)
    input_vocab = []
    output_vocab = []
    with open("data/input_vocab") as f:
        for line in f:
            input_vocab.append(line.strip())
        input_vocab = np.array(input_vocab)
    with open("data/output_vocab") as f:
        for line in f:
            output_vocab.append(line.strip())
        output_vocab = np.array(output_vocab)
    np.savez("data/SAOKE_DATA.npz", input_vocab=input_vocab, output_vocab=output_vocab,
             knowledge_train=knowledge_train,
             natural_train=natural_train,
             knowledge_test=knowledge_test,
             natural_test=natural_test, allow_pickle=True)


# tokenize_vocab()
# tokenize_id()
save_as_npy_by_train_test()
