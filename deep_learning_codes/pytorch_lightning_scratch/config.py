# -*- coding: utf-8 -*-
"""
Created on 2024-06-22 11:07:33

@author: borisσ, Chairman of FrameX Inc.

Our mission is as same as xAi's, 'Understand the Universe'.

I am recently interested in Multimodal Learning.
"""

# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 6
LEARNING_RATE = 0.001

# Dataset
DATA_DIR = ""
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16
