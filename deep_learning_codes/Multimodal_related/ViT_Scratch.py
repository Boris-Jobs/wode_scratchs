# -*- coding: utf-8 -*-
"""
Created on 2024-06-21 22:49:23

@author: borisσ, Chairman of FrameX Inc.

Our mission is as same as xAi's, 'Understand the Universe'.

I am recently interested in Multimodal Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def image2emb_naive(Image, Patch_size, Weight):
    # Shadows name 'image' from outer scope 表示外面(if __name__ == '__main__':)有全局的image了，你还用image来做输入参数
    patch = F.unfold(Image, kernel_size=Patch_size, stride=Patch_size).transpose(-1, -2)
    patch_embedding = patch @ Weight
    print(patch.shape)
    print('weight_size', Weight.shape)
    print(patch_embedding.shape)
    return patch_embedding


def image2emb_conv(image, kernel, stride):
    conv_output = F.conv2d(image, kernel, stride)
    conv_output.reshape()
    return conv_output


if __name__ == '__main__':
    bs, ic, image_h, image_w = 1, 3, 8, 8
    patch_size = 4
    model_dim = 8
    patch_depth = patch_size*patch_size*ic
    Image = torch.randn(bs, ic, image_h, image_w)
    weight = torch.randn(patch_depth, model_dim)

    patch_embedding_naive = image2emb_naive(Image, patch_size, weight)
