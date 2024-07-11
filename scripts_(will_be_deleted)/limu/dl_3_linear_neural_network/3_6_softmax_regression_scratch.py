# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:36:49 2024

@author: borisσ
"""

import torch
from IPython import display
from d2l import torch as d2l
import matplotlib.pyplot as plt


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

if __name__ == '__main__':

    X = torch.normal(0, 1, (2, 5))
    X_prob = softmax(X)
    X_prob, X_prob.sum(1)
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    y_hat[[0, 1], y]
    cross_entropy(y_hat, y)

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(
        num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))
    # keepdim的值：0就是把所有行压在一起，1就是把所有列压在一起
    # 对比学习：torch.cat([tensor1, tensor2], dim=1)


    print("accuracy: ", d2l.accuracy(y_hat, y) / len(y))
    
    
    lr = 0.1
    
    
    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.plt.show()
    d2l.evaluate_accuracy(net, test_iter)

"""
遇到报错：
RuntimeError: DataLoader worker (pid(s) 29396, 27064, 3472, 18476) exited unexpectedly

解决方案：
if __name__ == '__main__':
"""

