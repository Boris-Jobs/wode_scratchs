# -*- coding: utf-8 -*-
"""
Created on 2024-05-20 17:15:55

@author: Boris Jobs, Chairman of FrameX Inc.

Our mission is as same as xAi's, 'Understand the Universe'.

AI that benefits all humanity is all you need.
"""

import collections
import math
import torch
import _wode_functions as cz
from torch import nn, rnn

class Seq2SeqEncoder(cz.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)

        output, state = self.rnn(X)

        return output, state

