# -*- coding: utf-8 -*-
"""
Created on 2024-04-14 16:43:02

@author: Boris Jobs, Chairman of FrameX Inc.

Our mission is as same as xAi's, 'Understand the Universe'.

AI that benefits all humanity is all you need.

AlexNet以Alex Krizhevsky的名字命名，他是论文 (Krizhevsky et al., 2012)的第一作者。


"""

# (input.size()[0] - kernel_size + padding * 2) // stride + 1 = output.size()[0]

import torch
from torch import nn
import _wode_functions as cz

if __name__ == '__main__':
    net = nn.Sequential(
        # 这里使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet

        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
        # Conv2d output shape:	 torch.Size([1, 96, 54, 54])
        # (224 + 1 * 2 - 11) // 4 + 1 = 54 是不是丢掉了最右侧的3列信息，因为kernel走到第223列就没往右边走了
        nn.ReLU(),
        # ReLU output shape:	 torch.Size([1, 96, 54, 54])
        # ReLU()不改变tensor大小
        nn.MaxPool2d(kernel_size=3, stride=2),
        # MaxPool2d output shape:	 torch.Size([1, 96, 26, 26])
        # (54 - 3) // 2 + 1 = 26
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        # Conv2d output shape:	 torch.Size([1, 256, 26, 26])
        # (26 - 5 + 2 * 2) // 1 + 1 = 26
        nn.ReLU(),
        # ReLU output shape:	 torch.Size([1, 256, 26, 26])
        nn.MaxPool2d(kernel_size=3, stride=2),
        # MaxPool2d output shape:	 torch.Size([1, 256, 12, 12])
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        # Conv2d output shape:	 torch.Size([1, 384, 12, 12])
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        # Conv2d output shape:	 torch.Size([1, 384, 12, 12])
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        # Conv2d output shape:	 torch.Size([1, 256, 12, 12])
        nn.MaxPool2d(kernel_size=3, stride=2),
        # MaxPool2d output shape:	 torch.Size([1, 256, 5, 5])
        nn.Flatten(),
        # Flatten output shape:	 torch.Size([1, 6400])
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        ####### 为什么会需要一个Dropout在这里呢 #######
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(4096, 10))

    # Here is an example for tensor 'X':
    X = torch.randn(1, 1, 224, 224)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    batch_size = 128
    train_iter, test_iter = cz.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.01, 10
    cz.train_ch6(net, train_iter, test_iter, num_epochs, lr, cz.try_gpu())

