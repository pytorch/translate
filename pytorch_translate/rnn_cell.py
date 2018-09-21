#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def LSTMCell(input_dim, hidden_dim, **kwargs):
    m = nn.LSTMCell(input_dim, hidden_dim, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


class MILSTMCellBackend(nn.RNNCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MILSTMCellBackend, self).__init__(input_size, hidden_size, bias=False)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter("bias", None)
        self.alpha = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.beta_h = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.beta_i = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        # get prev_t, cell_t from states
        hx, cx = hidden
        Wx = F.linear(x, self.weight_ih)
        Uz = F.linear(hx, self.weight_hh)

        # Section 2.1 in https://arxiv.org/pdf/1606.06630.pdf
        gates = self.alpha * Wx * Uz + self.beta_i * Wx + self.beta_h * Uz + self.bias

        # Same as LSTMCell after this point
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy


def MILSTMCell(input_dim, hidden_dim, **kwargs):
    m = MILSTMCellBackend(input_dim, hidden_dim, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


class LayerNormLSTMCellBackend(nn.LSTMCell):
    def __init__(self, input_dim, hidden_dim, bias=True, epsilon=0.00001):
        super(LayerNormLSTMCellBackend, self).__init__(input_dim, hidden_dim, bias)
        self.epsilon = epsilon

    def _layerNormalization(self, x):
        mean = x.mean(1, keepdim=True).expand_as(x)
        std = x.std(1, keepdim=True).expand_as(x)
        return (x - mean) / (std + self.epsilon)

    def forward(self, x, hidden):
        hx, cx = hidden
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(
            hx, self.weight_hh, self.bias_hh
        )

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(self._layerNormalization(ingate))
        forgetgate = F.sigmoid(self._layerNormalization(forgetgate))
        cellgate = F.tanh(self._layerNormalization(cellgate))
        outgate = F.sigmoid(self._layerNormalization(outgate))

        cy = (forgetgate * cx) + (ingate * cellgate)

        hy = outgate * F.tanh(cy)

        return hy, cy


def LayerNormLSTMCell(input_dim, hidden_dim, **kwargs):
    m = LayerNormLSTMCellBackend(input_dim, hidden_dim, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m
