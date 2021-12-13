# -*- encoding: utf-8 -*-
"""
@File    : datasets.py
@Time    : 29/11/21 8:24 pm
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from config import Config
from utils.prepare_data import Vocab


class TextDataset(Dataset):
    def __init__(self, config, vocab, mode="train", num_data=None):
        self.mode = mode
        self.input_max_len = config.input_max_len
        self.output_max_len = config.output_max_len
        self.device = config.device
        self.vocab = vocab

        if mode == "train":
            self.train_x = []
            self.train_y = []
            with open(config.train_x_path, "r") as f:
                data_list = f.readlines()
                if num_data is not None:
                    data_list = data_list[:num_data]
                for line in tqdm(data_list):
                    line = [s for s in line.strip().split(" ") if s != ""]
                    if len(line) < self.input_max_len:
                        line += ["<PAD>"] * (self.input_max_len - len(line))
                    else:
                        line = line[:self.input_max_len]
                    self.train_x.append(line)

            with open(config.train_y_path, "r") as f:
                data_list = f.readlines()
                if num_data is not None:
                    data_list = data_list[:num_data]
                for line in tqdm(data_list):
                    line = [s for s in line.strip().split(" ") if s != ""]
                    if len(line) < self.output_max_len:
                        line += ["<END>"] + ["<PAD>"] * (self.output_max_len - len(line) - 1)
                    else:
                        line = line[:self.output_max_len]
                    self.train_y.append(line)

            self.size = len(self.train_x)

        elif mode == "eval":
            self.eval_x = []
            self.eval_y = []
            #             self.size = len(self.train_x)
            with open(config.eval_x_path, "r") as f:
                data_list = f.readlines()
                if num_data is not None:
                    data_list = data_list[:num_data]
                for line in tqdm(data_list):
                    line = [s for s in line.strip().split(" ") if s != ""]
                    if len(line) < self.input_max_len:
                        line += ["<PAD>"] * (self.input_max_len - len(line))
                    else:
                        line = line[:self.input_max_len]
                    self.eval_x.append(line)

            with open(config.eval_y_path, "r") as f:
                data_list = f.readlines()
                if num_data is not None:
                    data_list = data_list[:num_data]
                for line in tqdm(data_list):
                    line = [s for s in line.strip().split(" ") if s != ""]
                    self.eval_y.append(line)
            self.size = len(self.eval_x)

        else:
            self.test_x = []
            with open(config.test_x_path, "r") as f:
                for line in tqdm(f.readlines()[:num_data]):
                    line = [s for s in line.strip().split(" ") if s != ""]
                    if len(line) < self.input_max_len:
                        line += ["<PAD>"] * (self.input_max_len - len(line))
                    else:
                        line = line[:self.input_max_len]
                    self.test_x.append(line)
            self.size = len(self.test_x)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.mode == "train":
            train_x_tokens = []
            for x in self.train_x[idx]:
                if x not in self.vocab.word2id:
                    train_x_tokens.append(self.vocab.word2id["<UNK>"])
                else:
                    train_x_tokens.append(self.vocab.word2id[x])

            train_y_tokens = []
            for x in self.train_y[idx]:
                if x not in self.vocab.word2id:
                    train_y_tokens.append(self.vocab.word2id["<UNK>"])
                else:
                    train_y_tokens.append(self.vocab.word2id[x])

            return torch.tensor(train_x_tokens).to(self.device), torch.tensor(train_y_tokens).to(self.device)

        elif self.mode == "eval":
            eval_x_tokens = []
            for x in self.eval_x[idx]:
                if x not in self.vocab.word2id:
                    eval_x_tokens.append(self.vocab.word2id["<UNK>"])
                else:
                    eval_x_tokens.append(self.vocab.word2id[x])

            return torch.tensor(eval_x_tokens).to(self.device), self.eval_y[idx]

        else:
            test_x_tokens = []
            for x in self.test_x[idx]:
                if x not in self.vocab.word2id:
                    test_x_tokens.append(self.vocab.word2id["<UNK>"])
                else:
                    test_x_tokens.append(self.vocab.word2id[x])
            return torch.tensor(test_x_tokens).to(self.device)


if __name__ == '__main__':
    config = Config()
    vocab = Vocab()
    vocab.reload(config.vocab_path)
    train_dataset = TextDataset(config, vocab, "train",100)
    test_dataset = TextDataset(config, vocab, "test",50)
    eval_dataset = TextDataset(config, vocab, "eval",10)

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    train_x, train_y = next(iter(train_dataloader))
    print(train_x)
    print(train_y)