""" Implementation of ONMT RNN for Input Feeding Decoding """
import torch
import torch.nn as nn
import math
import torch as th
import logging


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.rnn_size = rnn_size

        for _ in range(num_layers):
            self.layers.append(LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        if hidden is None:
            if torch.cuda.is_available():
                hidden = (torch.zeros(self.num_layers, input_feed.shape[1], self.rnn_size).cuda(),
                          torch.zeros(self.num_layers, input_feed.shape[1], self.rnn_size).cuda())
            else:
                 hidden = (torch.zeros(self.num_layers, input_feed.shape[1], self.rnn_size),
                          torch.zeros(self.num_layers, input_feed.shape[1], self.rnn_size))


        # logging.info(input_feed.shape)
        # logging.info(hidden[0].shape)
        h_0, c_0 = hidden
        h_1, c_1, i_1, f_1, o_1 = [], [], [], [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i, i_1_i, f_1_i, o_1_i = layer(
                input_feed, (h_0[i].unsqueeze(0), c_0[i].unsqueeze(0)))
            input_feed = h_1_i
            if i + 1 != self.num_layers and self.training:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i.squeeze(0)]
            c_1 += [c_1_i.squeeze(0)]
            i_1 += [i_1_i.squeeze(0)]
            f_1 += [f_1_i.squeeze(0)]
            o_1 += [o_1_i.squeeze(0)]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        i_1 = torch.stack(i_1)
        f_1 = torch.stack(f_1)
        o_1 = torch.stack(o_1)
        return input_feed, (h_1, c_1), i_1, f_1, o_1


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        # std = 1.0 / math.sqrt(self.hidden_size)
        # for w in self.parameters():
        #    w.data.uniform_(-std, std)
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self._init_hidden(x)

        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)
        # logger.info(preact.shape)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        h_t = th.mul(o_t, c_t.tanh())

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)

        i_t = i_t.view(1, h.size(0), -1)
        f_t = f_t.view(1, h.size(0), -1)
        o_t = o_t.view(1, h.size(0), -1)
        return h_t, c_t, i_t, f_t, o_t

    @staticmethod
    def _init_hidden(input_):
        h = th.zeros_like(input_.view(1, input_.size(1), -1))
        c = th.zeros_like(input_.view(1, input_.size(1), -1))
        return h, c


class StackedGRU(nn.Module):
    """
    Our own implementation of stacked GRU.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input_feed, hidden[0][i])
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input_feed, (h_1,)
