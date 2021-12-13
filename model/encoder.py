# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import DataLoader

from config import Config
from utils.prepare_data import Vocab


class Encoder(torch.nn.Module):
    def __init__(self, config, vocab):
        super(Encoder, self).__init__()
        self.vocab = vocab
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.batch_size = config.batch_size

        self.embedding = torch.nn.Embedding(len(self.vocab), self.input_size)

        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers
                                  , batch_first=True, bidirectional=self.bidirectional)

        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, inputs, hidden=None, cell=None):
        """
        inputs: batch_size, seq_len
        init_hidden/init_cell: numb_layers * 2, batch_size, hidden_size
        """
        inputs = self.embedding(inputs)
        # inputs after embedding: batch_size, seq_len, input_size
        if hidden is not None:
            outputs, (hidden, cell) = self.lstm(inputs, (hidden, cell))
        else:
            outputs, (hidden, cell) = self.lstm(inputs)
        return outputs, hidden, cell

    def init_hidden_cell(self):
        D = self.num_layers * 2 if self.bidirectional else self.num_layers
        return torch.zeros(D, self.batch_size, self.hidden_size), torch.zeros(D, self.batch_size, self.hidden_size)

if __name__ == '__main__':
    config = Config()
    vocab = Vocab()
    vocab.reload(config.vocab_path)

    from utils.datasets import TextDataset
    dataset = TextDataset(config, vocab, "train", 10)
    dataloader = DataLoader(dataset, 4, True)

    train_x, train_y = next(iter(dataloader))

    encoder = Encoder(config, vocab)
    outputs, hidden, cell = encoder(train_x)
    print(f"outputs shape {outputs.shape}")  # 4, 50 ,128 batch_size, input_len, 2 * hidden_size
    print(f"hidden shape {hidden.shape}")  # 2 4 64 num_layer*bidi, batch_size, hidden_size
    print(f"cell shape {cell.shape}")  # 2 4 64 num_layer*bidi, batch_size, hidden_size