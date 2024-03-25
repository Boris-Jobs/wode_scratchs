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
    random.shuffle()


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    test_tensor = torch.tensor([[[5, 3, 4], [2, 3, 4]],[[2, 3, 4],[3, 4, 5]]])

    print(labels[1: 10], true_w.shape, labels.shape, test_tensor.shape)