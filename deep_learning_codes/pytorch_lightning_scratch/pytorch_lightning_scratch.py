# -*- coding: utf-8 -*-
"""
Created on 2024-06-21 11:38:16

@author: borisσ, Chairman of FrameX Inc.

Our mission is as same as xAi's, 'Understand the Universe'.

I am recently interested in Multimodal Learning.
"""

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torch.optim import Adam
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
    def __init__(self, inputSize, numClasses):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, 50)
        self.fc2 = nn.Linear(50, numClasses)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=numClasses)
        self.my_accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=numClasses)

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
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
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


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, bs, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = bs
        self.num_workers = num_workers

    def prepare_data(self):
        # single GPU
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        # multi GPU
        entire_dataset = datasets.MNIST(root=self.data_dir, train=True, transform=transforms.ToTensor(), download=False)
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
        self.test_ds = datasets.MNIST(root=self.data_dir, train=False, transform=transforms.ToTensor(), download=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    input_size = 784
    learning_rate = 0.001
    num_classes = 10
    num_epochs = 3

    model = NN(inputSize=input_size, numClasses=num_classes).to(device)
    trainer = pl.Trainer(accelerator='gpu', devices=[0], min_steps=1, max_steps=3, precision=16)
    dm = MnistDataModule(data_dir='', bs=batch_size, num_workers=8)

    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=dm)



