# -*- encoding: utf-8 -*-
"""
@File    : config.py
@Time    : 26/11/21 6:02 pm
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import torch

class Config():
    def __init__(self):
        self.app_dir = "/Users/duxy/Downloads/PycharmProjects/TextSum"
        self.data_dir = self.app_dir + "/data"
        self.out_dir = self.app_dir + "/outs"
        self.log_dir = self.app_dir + "/logs"
        self.model_name = "Seq2Seq_Attention"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.train_x_path = self.data_dir + "/train_x.txt"
        self.train_y_path = self.data_dir + "/train_y.txt"
        self.test_x_path = self.data_dir + "/train_x.txt"
        self.eval_x_path = self.data_dir + "/train_x.txt"
        self.eval_y_path = self.data_dir + "/train_y.txt"

        self.paths_list = [self.train_x_path, self.train_y_path,
                          self.test_x_path,
                          self.eval_x_path, self.eval_y_path]

        self.vocab_path = self.data_dir + "/vocab.txt"

        # training
        self.epoch_num = 1
        self.lr = 0.01
        self.input_size = 200
        self.hidden_size = 64
        self.dropout = 0.3
        self.num_layers = 1
        self.bidirectional = True
        self.batch_size = 16
        self.input_max_len = 256
        self.output_max_len = 64
