# -*- coding: utf-8 -*-
"""
Created on 2024-06-22 11:36:31

@author: borisσ, Chairman of FrameX Inc.

Our mission is as same as xAi's, 'Understand the Universe'.

I am recently interested in Multimodal Learning.
"""

from pytorch_lightning.callbacks import EarlyStopping, Callback

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module) -> None:
        print("@boris, starting to train.")

    def on_train_end(self, trainer, pl_module) -> None:
        print("@boris, training is done.")

