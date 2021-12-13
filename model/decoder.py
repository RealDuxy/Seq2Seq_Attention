# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import DataLoader

from config import Config
from model.encoder import Encoder
from utils.prepare_data import Vocab


class Decoder(torch.nn.Module):
    def __init__(self, config, vocab):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.input_size = config.input_size
        self.output_size = len(vocab)
        self.hidden_size = config.hidden_size
        self.dropout = torch.nn.Dropout(config.dropout)
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.input_max_len = config.input_max_len

        self.embedding = torch.nn.Embedding(len(self.vocab), self.input_size)
        self.lstm = torch.nn.LSTM(self.input_size + self.hidden_size, self.hidden_size, num_layers=self.num_layers
                                  , batch_first=True, bidirectional=self.bidirectional)
        self.fc = torch.nn.Linear(self.hidden_size * 2, self.output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden, cell, context_vector):
        """
        inputs: batch_size, input_len
        hidden: batch_size, 1 , hidden_size
        cell: batch_size, 1 , hidden_size
        context_vector: batch_size, 1, hidden_size
        """
        # 获取embedding
        inputs = self.embedding(inputs)
        # after inputs: batch_size, 1, input_size

        # 与context_vector concat, 在这里结合注意力信息
        inputs = torch.cat((inputs, context_vector), 2)
        # after inputs: batch_size, 1, input_size+hidden_size

        # 进入lstm模块
        outputs, (hidden, cell) = self.lstm(inputs, (hidden, cell))
        # outputs: batch_size, 1, hidden_size

        # 利用lstm的outputs进行预测输出当前token prob distribution
        prediction = self.softmax(self.fc(torch.squeeze(outputs, 1)))

        return prediction, outputs, hidden, cell

if __name__ == '__main__':
    config = Config()
    vocab = Vocab()
    vocab.reload(config.vocab_path)
    batch_size = 4

    from utils.datasets import TextDataset
    dataset = TextDataset(config, vocab, "train", 10)
    dataloader = DataLoader(dataset, batch_size, True)

    train_x, train_y = next(iter(dataloader))

    encoder = Encoder(config, vocab)

    outputs, hidden, cell = encoder(train_x)
    print(f"outputs shape {outputs.shape}")  # 4, 50 ,128 batch_size, input_len, 2 * hidden_size
    print(f"hidden shape {hidden.shape}")  # 2 4 64 num_layer*bidi, batch_size, hidden_size
    print(f"cell shape {cell.shape}")  # 2 4 64 num_layer*bidi, batch_size, hidden_size

    context_vector = torch.zeros(batch_size, 1, 64)

    decoder = Decoder(config, vocab)

    decoder_input = torch.tensor([vocab.word2id["<START>"]] * batch_size).view(-1, 1)

    prediction, outputs, hidden, cell = decoder(decoder_input, hidden, cell, context_vector)
    print(f"prediction shape {prediction.shape}")  # 4 37779 batch_size, out_size
    print(f"outputs shape {outputs.shape}")  # 4 1 128 batch_size , 1, 2 * hidden
    print(f"hidden shape {hidden.shape}")  # 2 4 64 num_layer*bidi, batch_size, hidden_size
