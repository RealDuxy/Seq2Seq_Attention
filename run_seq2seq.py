# -*- encoding: utf-8 -*-
"""
@File    : run_seq2seq.py
@Time    : 12/12/21 8:29 pm
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import torch
from torch.utils.data import DataLoader

from config import Config
from utils.prepare_data import Vocab
from model.seq2seq import Seq2Seq
from utils.train_eval_test_helper import train, evaluate, predict
from utils.datasets import TextDataset


def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    vocab = Vocab()
    vocab.reload(config.vocab_path)
    model = Seq2Seq(config, vocab).to(device)

    train_dataset = TextDataset(config, vocab, "train",30)
    eval_dataset = TextDataset(config, vocab, "eval", 10)
    test_dataset = TextDataset(config, vocab, "test",10)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train(model, train_dataloader, eval_dataloader, config)
    evaluate(model, eval_dataloader)
    predict(model, test_dataloader, config.out_dir + "/Seq2Seq/test_predictions.txt")

if __name__ == '__main__':
    main()