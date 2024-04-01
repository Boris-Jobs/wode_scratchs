# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 23:16:02 2024

@author: borisσ
"""



import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
d2l.use_svg_display()


trans = transforms.ToTensor()
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间

mnist_train = torchvision.datasets.FashionMNIST(
    root=".", train=True, transform=trans, download=True)

mnist_test = torchvision.datasets.FashionMNIST(
    root=".", train=False, transform=trans, download=True)

# Fashion-MNIST训练集和测试集分别包含60000和10000张图像

print('每张图的大小：', mnist_train[0][0].shape)

print('训练集和测试集大小：', len(mnist_train), len(mnist_test))


def get_fashion_mnist_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress',
                   'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.detach().numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])

    return axes


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));

# 检查是否可用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# 将数据移动到 GPU 上
X = X.to(device)
y = y.to(device)


batch_size = 256


def get_dataloader_workers():
    return 4


# 使用4个进程来读取数据
train_iter = data.DataLoader(
    mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())


def load_data_fashion_mnist(batch_size, resize=None):  # @save

    # 下载Fashion-MNIST数据集，然后将其加载到内存中
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=".", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=".", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    X = X.to(device)
    y = y.to(device)
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
