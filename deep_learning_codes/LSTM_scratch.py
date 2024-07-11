# -*- coding: utf-8 -*-
"""
Created on 2024-07-11 18:39:58

@author: borisσ, Chairman of FrameX Inc.

Our mission is as same as xAi's, 'Understand the Universe'.

I am recently interested in Multimodal Learning.
"""

import torch
from torch import nn
from _wode_functions import torch as cz

batch_size, num_steps = 32, 35
train_iter, vocab = cz