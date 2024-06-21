# -*- coding: utf-8 -*-
"""
Created on 2024-06-21 11:38:16

@author: borisσ, Chairman of FrameX Inc.

Our mission is as same as xAi's, 'Understand the Universe'.

I am recently interested in Multimodal Learning.
"""

import torch
import pytorch_lightning as pl
from blackd.middlewares import F
from torchvision.datasets import MNIST
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy
import torchmetrics
from torchmetrics import Metric

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()


class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.my_accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        my_accuracy = self.my_accuracy(scores, y)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score,
                      'train_my_accuracy': my_accuracy}, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'accuracy': accuracy, 'f1_score': f1_score, 'my_accuracy': my_accuracy}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._test_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._test_step(batch, batch_idx)
        self.log('test_loss', loss)
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




if __name__ == '__main__':
    trainer = pl.Trainer(accelerator='gpu', devicees=[0], min_steps=1, max_steps=3, precision=16,
                         num_nodes=1)

