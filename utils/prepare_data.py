# -*- encoding: utf-8 -*-
"""
@File    : prepare_data.py
@Time    : 25/11/21 10:38 am
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import pandas as pd
import jieba
import numpy as np
from tqdm import tqdm
import torch
from config import Config

from collections import Counter

config = Config()

class Vocab():
    def __init__(self):
        self.id2word = {0: "<UNK>", 1: "<START>", 2: "<END>", 3: "<PAD>"}
        self.word2id = {"<UNK>": 0, "<START>": 1, "<END>": 2, "<PAD>": 3}
        #         self.word_count = {"UNK":0}
        self.size = 5


    def add(self, word):
        self.id2word[self.size] = word
        self.word2id[word] = self.size
        self.size += 1

    def __len__(self):
        return self.size

    def save(self, path=config.vocab_path):
        with open(path, "w") as f:
            for word, idx in self.word2id.items():
                f.write(f"{word} {idx}\n")
        print(f"vocab saved in {path}, vocab size: {self.size}")

    def reload(self, path=config.vocab_path):
        with open(path, "r") as f:
            idx = 1
            for line in f.readlines():
                sentence = line.strip().split()
                word = str(sentence[0])
                idx = int(sentence[1])
                self.id2word[idx] = word
                self.word2id[word] = idx
        self.size = len(self.id2word)
        print(f"vocab loaded from {path}")


def build_vocab(datas_list, threshold=0.9):
    "统计词频，根据给定的阈值筛选后，保存至词典"
    all_context_list = []
    for x in datas_list:
        for y in x:
            for word in y:
                all_context_list.append(word)

    cnt = Counter(all_context_list)
    all_words_num = len(cnt)
    words_num_after = int(threshold * all_words_num)
    print(
        f"all words num {all_words_num}, words after {words_num_after}, {all_words_num - words_num_after} were throwed")

    vocab = Vocab()
    for char, freq in cnt.most_common(words_num_after):
        if freq > 3 and char != "":
            vocab.add(char)
    print(f"vocab size {len(vocab)}")
    vocab.save()
    return vocab

def build_train_test_val_list(train_df, test_df):
    num_val = int(len(train_df) * 0.1)
    num_train = len(train_df) - num_val
    num_test = len(test_df)

    def df2list(df, num):
        # train or val
        if df.shape[1] == 6:
            # train
            if num > 0:
                x_df = df.iloc[:num, 1].map(str) + "/"
                for i in range(2, 5):
                    x_df += "/" + df.iloc[:num, i].map(str)
                y_df = df.iloc[:num,5]
            else: # val
                x_df = df.iloc[num:, 1].map(str) + "/"
                for i in range(2, 5):
                    x_df += "/" + df.iloc[num:, i].map(str)
                y_df = df.iloc[num:,5]
            return x_df.tolist(), y_df.tolist()
        else: # test
            x_df = df.iloc[:num, 1].map(str) + "/"
            for i in range(2, 5):
                x_df += "/" + df.iloc[:num, i].map(str)
            return x_df.tolist()

    train_x_list, train_y_list = df2list(train_df, num_train)
    test_x_list = df2list(test_df, num_test)
    val_x_list, val_y_list = df2list(train_df, -num_val)

    return [train_x_list, train_y_list, test_x_list, val_x_list, val_y_list]

def save(data_list, data_path):
    with open(data_path, "w") as f:
        for sentence in tqdm(data_list):
            sentence = str(sentence)
            word_list = jieba.lcut(sentence.strip())
            for word in word_list:
                f.write(word.strip() + " ")
            f.write("\n")
    print(f"{data_path} saved")




def load_train_test_val(path_list):
    datas_list = []
    for data_path in path_list:
        data_list = []
        with open(data_path, "r") as f:
            for line in tqdm(f.readlines()):
                data_list.append(line.strip().split(" "))
            datas_list.append(data_list)
        print(f"{data_path} loaded")
    return datas_list

if __name__ == '__main__':
    config = Config()
    train_df = pd.read_csv("../data/AutoMaster_TrainSet.csv").dropna(how="all")
    test_df = pd.read_csv("../data/AutoMaster_TestSet.csv").dropna(how="all")

    datas_list = build_train_test_val_list(train_df, test_df)
    for i in range(len(datas_list)):
        data_list = datas_list[i]
        data_path = config.paths_list[i]
        save(data_list, data_path)
    datas_list = load_train_test_val(config.paths_list)
    build_vocab(datas_list)




