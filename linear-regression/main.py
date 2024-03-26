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

def example_generate_batch():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    batch_size = 10
    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        with torch.no_grad():
            param -= lr * param.grad / batch_size
            param.grad.zero_()



def example_sgd():
    lr = 0.03
    num_epochs = 3
    batch_size = 10
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    features, labels = synthetic_data(w, b, 1000)
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = squared_loss(linreg(X, w, b), y)
            l.sum().backward(retain_graph=True)  # 为什么加了一个retain_graph就不会再报错了？
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = squared_loss(linreg(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

if __name__ == '__main__':
    example_generate_batch()
    example_sgd()

    """

    test_tensor = torch.tensor([[[5, 3, 4], [2, 3, 4]],[[2, 3, 4],[3, 4, 5]]])

    print(labels[1: 10], true_w.shape, labels.shape, test_tensor.shape)    
    """
