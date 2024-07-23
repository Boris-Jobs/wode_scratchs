# srun -N 1 -n 1 -c 3 -t 00:10:00 --mem-per-cpu=32G --gres=gpu:a100:2 -p gputest --account=project_2006362 python3 ./scratchs/multi_GPU_scratch.py

import torch
from torch import nn
from torch.nn import functional as F
import _wode_functions as CZ
from time import sleep

sleep(600)

def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat


if __name__ == '__main__':
    scale = 0.01
    W1 = torch.randn(size=(20, 1, 3, 3)) * scale
    b1 = torch.zeros(20)
    W2 = torch.randn(size=(50, 20, 5, 5)) * scale
    b2 = torch.zeros(50)
    W3 = torch.randn(size=(800, 128)) * scale
    b3 = torch.zeros(128)
    W4 = torch.randn(size=(128, 10)) * scale
    b4 = torch.zeros(10)
    params = [W1, b1, W2, b2, W3, b3, W4, b4]

    loss = nn.CrossEntropyLoss(reduction='none')