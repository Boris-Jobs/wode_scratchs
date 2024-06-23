# -*- coding: utf-8 -*-
"""
Created on 2024-06-22 11:10:05

@author: borisσ, Chairman of FrameX Inc.

Our mission is as same as xAi's, 'Understand the Universe'.

I am recently interested in Multimodal Learning.
"""

import torch
import pytorch_lightning as pl
from model import NN
from dataset import MnistDataModule
import config
from Callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("medium")  # to make lightning happy

if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="boris_model")
    model = NN(inputSize=config.INPUT_SIZE, numClasses=config.NUM_CLASSES)
    dm = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_CLASSES,
    )
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=3,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")]
    )
    trainer.fit(model, datamodule=dm)
    trainer.validate(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)
