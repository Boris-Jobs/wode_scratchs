import random
import torch


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 给y制造噪音，均值为0标准差为0.01的正态分布噪音
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 把索引进行洗牌，随机挑选batch
    for i in range(0, num_examples, batch_size):
        # batch_indices = indices[i: min(i + batch_size, num_examples)]
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])  # 把batch_indices做成张量也可以。
        yield features[batch_indices], labels[batch_indices]



def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        # 一旦.backward()以后，叶子节点的值就不可以变化了，得有torch.no_grad()才行
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            # 不添加zero这一条的话会导致新来的

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


lr = 20000000
num_epochs = 10
batch_size = 100
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
features, labels = synthetic_data(w, b, 1000)
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = squared_loss(linreg(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        print(f'w = {w}, b = {b}')
        l.sum().backward(retain_graph=True)
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        print(f'w = {w}, b = {b}')
        break
    train_l = squared_loss(linreg(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    
    



def example_generate_batch():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    batch_size = 10
    for i in range(10):
        for X, y in data_iter(batch_size, features, labels):
            print(X, '\n', y)
            break


def example_sgd():
    lr = 2000000
    num_epochs = 20
    batch_size = 1000
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    features, labels = synthetic_data(w, b, 10000)
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = squared_loss(linreg(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            print(f'w = {w}, b = {b}')
            l.sum().backward(retain_graph=True)
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
            print(f'w = {w}, b = {b}')
            break
        train_l = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')







if __name__ == '__main__':
    # example_generate_batch()
    # example_sgd()

    """

    test_tensor = torch.tensor([[[5, 3, 4], [2, 3, 4]],[[2, 3, 4],[3, 4, 5]]])

    print(labels[1: 10], true_w.shape, labels.shape, test_tensor.shape)    
    """


"""

import torch

x = torch.tensor([[2., 4.], [3., 1.], [5., 6.], [7., 8.], [9., 2.]], requires_grad=True)
w = torch.tensor([[1.], [4.]], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)
y = torch.tensor([[1.], [2.], [3.], [4.], [5.]])

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


l = squared_loss(linreg(x, w, b), y)
l.sum().backward(retain_graph=True)

print(x.grad, '\n', w.grad, '\n', b.grad)

with torch.no_grad():
    for param in [x,w,b]:
        param -= param.grad
        param.grad.zero_()

l = squared_loss(linreg(x, w, b), y)
l.sum().backward(retain_graph=True)

print(x.grad, '\n', w.grad, '\n', b.grad)



"""