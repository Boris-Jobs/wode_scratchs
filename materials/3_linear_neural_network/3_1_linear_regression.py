# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:03:15 2024

@author: borisσ
"""

"""

%matplotlib inline

%matplotlib inline is a magic command in Jupyter Notebooks or IPython that allows plots to be displayed directly in the notebook. 

"""

import math
import time
import numpy as np
import torch
from d2l import torch as d2l



class Timer: #@save

    def __init__(self):
        self.times = []
        self.start()
        
    def start(self):
        self.tik = time.time()
        
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times) / len(self.times)
    
    def sum(self):
        return sum(self.times)
    
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
 
    
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)
    


if __name__ == "__main__":
    
    
    n = 10000
    a = torch.ones([n])
    b = torch.ones([n])
    
    c = torch.zeros(n)
    timer = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f'first method: {timer.stop():.5f} sec')
    
    # 矢量化加速：在训练我们的模型时，我们经常希望能够同时处理整个小批量的样本。为了实现这一点，需要我们对计算进
    # 行矢量化，从而利用线性代数库，而不是在Python中编写开销高昂的for循环。
    timer.start()
    d = a + b
    print(f'second method: {timer.stop():.25f} sec')
    
    
    # 正态分布与平方损失
    
    x = np.arange(-7, 7, 0.01)
    
    params = [(0, 1), (0, 2), (3, 1)]
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
             ylabel='p(x)', figsize=(4.5, 2.5), 
             legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])

    