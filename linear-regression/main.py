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
    random.shuffle(indices)  # 把索引进行洗牌
    for i in range(0, num_examples, batch_size):
        # batch_indices = indices[i: min(i + batch_size, num_examples)]
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])  # 把batch_indices做成张量也可以。
        yield features[batch_indices], labels[batch_indices]



if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    batch_size = 10
    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break

    """

    test_tensor = torch.tensor([[[5, 3, 4], [2, 3, 4]],[[2, 3, 4],[3, 4, 5]]])

    print(labels[1: 10], true_w.shape, labels.shape, test_tensor.shape)    
    """
