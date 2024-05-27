# -*- coding: utf-8 -*-
"""
Created on 2024-05-27 18:52:56

@author: borisσ, Chairman of FrameX Inc.

Our mission is as same as xAi's, 'Understand the Universe'.

I am recently interested in Multimodal Learning.
"""

import torch
from torch import nn
import _wode_functions as cz

class AttentionDecoder(cz.Decoder):  # 带有注意力机制解码器的基本接口
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        self.something = vocab_size


if __name__ == '__main__':
    a = 1
