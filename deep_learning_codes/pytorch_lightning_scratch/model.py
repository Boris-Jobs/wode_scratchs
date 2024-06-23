# -*- coding: utf-8 -*-
"""
Created on 2024-06-22 11:05:23

@author: borisσ, Chairman of FrameX Inc.

Our mission is as same as xAi's, 'Understand the Universe'.

I am recently interested in Multimodal Learning.
"""

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torchvision.utils
from torch.optim import Adam
import torchmetrics
from torchmetrics import Metric
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()


class NN(pl.LightningModule):
    def __init__(self, inputSize, numClasses):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, 50)
        self.fc2 = nn.Linear(50, numClasses)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=numClasses)
        self.my_accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=numClasses)
        self.epoch_y = []
        self.epoch_scores = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        my_accuracy = self.my_accuracy(scores, y)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
                "train_my_accuracy": my_accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("mnist_images", grid, self.global_step)
        self.epoch_scores.append(scores)
        self.epoch_y.append(y)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "my_accuracy": my_accuracy,
            "scores": scores
        }

    def on_train_epoch_end(self):
        # 将整个 epoch 的 scores 和 y 合并
        epoch_scores = torch.cat(self.epoch_scores)
        epoch_y = torch.cat(self.epoch_y)

        # 清空列表以备下一个 epoch 使用
        self.epoch_scores.clear()
        self.epoch_y.clear()

        # 计算 epoch 级别的指标
        train_acc = self.accuracy(epoch_scores, epoch_y)
        train_fi = self.f1_score(epoch_scores, epoch_y)

        # 使用 self.log 记录 epoch 级别的指标
        self.log_dict({
            "train_acc": train_acc,
            "train_fi": train_fi,
        }, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
