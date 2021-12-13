# -*- encoding: utf-8 -*-

import random

from torch.utils.data import DataLoader

from model.attention import Attention
from model.encoder import Encoder
from model.decoder import Decoder
from utils.prepare_data import Vocab
from config import Config
import torch


class Seq2Seq(torch.nn.Module):
    def __init__(self, config, vocab):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.vocab = vocab
        self.hidden_size = config.hidden_size
        self.input_max_len = config.input_max_len
        self.output_max_len = config.output_max_len
        self.attn = Attention(config)
        self.encoder = Encoder(config, vocab)
        self.decoder = Decoder(config, vocab)


    def forward(self, encoder_input, target=None, teacher_ratio=None):
        target_len = self.output_max_len
        batch_size = encoder_input.shape[0]

        outputs = torch.zeros(batch_size, target_len, len(self.vocab))

        decoder_input = torch.tensor([self.vocab.word2id["<START>"]] * batch_size).view(-1, 1)

        encoder_outputs, hiddens, cells = self.encoder(encoder_input)
        # ecncoder_outputs: batch_size, input_len, hidden_size

        # 初始化 context_vector
        context_vector = torch.zeros(batch_size, 1, self.hidden_size)

        for i in range(0, target_len):
            next_predictions, next_outputs, hiddens, cells = self.decoder(decoder_input, hiddens, cells, context_vector)
            outputs[:, i, :] = next_predictions

            # 计算attention weight
            attn_weight = self.attn(encoder_outputs, hiddens) # batch_size, input_len, 1
            # 与encoder outputs 矩阵相乘,得到context_vector
            temp_encoder_outputs = encoder_outputs.transpose(2, 1)  # batch_size, hidden_len * 2, input_len
            context_vector = torch.bmm(temp_encoder_outputs, attn_weight).transpose(2,1)
            # context vector: batch_size, 1, 2*hidden_state, 相加即可,获取到下一轮的context_vector
            context_vector = context_vector[:, :, :self.hidden_size] + context_vector[:,:,self.hidden_size:]

            # schedule sampling决定下一轮的的input
            if teacher_ratio is not None and random.random() < teacher_ratio:
                # 使用 teacher forcing
                decoder_input = target[:, i].unsqueeze(1)
            else:
                decoder_input = next_predictions.argmax(1).unsqueeze(1)

        return outputs


if __name__ == '__main__':
    config = Config()
    vocab = Vocab()
    vocab.reload(config.vocab_path)

    from utils.datasets import TextDataset

    dataset = TextDataset(config, vocab, "train", 10)
    dataloader = DataLoader(dataset, 4, True)

    train_x, train_y = next(iter(dataloader))
    model = Seq2Seq(config, vocab)
    outputs = model(train_x, train_y)

