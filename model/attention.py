# -*- encoding: utf-8 -*-
"""
@File    : attention.py
@Time    : 13/12/21 2:41 pm
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import torch
import torch.nn.functional as F


class Attention(torch.nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.hidden_size = config.hidden_size
        self.fc = torch.nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.v = torch.nn.Parameter(torch.randn(self.hidden_size))

    def forward(self, encoder_outputs, decoder_hidden):
        """
        得到context vector
        :param encoder_outputs: batch_size, input_len, 2 * hidden_size
        :param decoder_hidden: 2 4 64 num_layer*bidi, batch_size, hidden_size
        :return: (batch_size, 1, hidden_size)
        """
        # sum decoder bidi hidden
        attn_score = self.score(decoder_hidden, encoder_outputs)
        attn_weight = F.softmax(attn_score, 1)

        # attn_weight batch_size, input_len, 1
        return attn_weight

    def score(self, decoder_hidden, encoder_outputs):
        decoder_hidden_sum = torch.unsqueeze(decoder_hidden[0, :, :] + decoder_hidden[1, :, :], 1)
        # after unsqueeze decoder_hiddens_sum: batch_size, 1, hidden_size
        step = encoder_outputs.size(1)
        decoder_hidden_sum = decoder_hidden_sum.repeat((1, step, 1))
        # after repeat decoder_hidden: batch_size, seq_len, hidden_size

        # concat with encoder_outputs
        enc_dec_hidden = torch.cat((encoder_outputs, decoder_hidden_sum), 2)
        # enc_dec_hidden: batch_size, seq_len, hidden_size*2
        attn_weights = self.fc(enc_dec_hidden)
        attn_weights = torch.relu(attn_weights)
        # attn_weights: batch_size, seq_len, hidden_size
        v = self.v.repeat((attn_weights.shape[0], 1)).unsqueeze(2)  # v: batch_size, hidden_size, 1
        attn_score = torch.bmm(attn_weights, v)
        # atten_score: batch_size, seq_len, 1
        return attn_score

