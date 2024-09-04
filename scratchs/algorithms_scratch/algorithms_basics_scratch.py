# -*- coding: utf-8 -*-
"""
Created on 2024-09-02 10:22:08

@author: borisσ, Chairman of FrameX Inc.

I hope to use AI or LLMs to help people better understand the world and humanity.

We are big fans of xAI.

I am recently interested in Multimodal LLMs.
"""

if __name__ == "__main__":
    # 1. 以下两种矩阵初始化方式不同
    matrix1 = [[0] * 3] * 3
    matrix2 = [[0] * 3 for _ in range(3)]
    matrix1[0][0] = 1
    matrix2[0][0] = 1
    print("两个矩阵的第一列是不一样的: ", matrix1, matrix2)

    # 2. Python直接用元组解包来交换值
    a, b = 2, 3
    a, b = b, a

    # 3. 不等式可以连写
    if a == b == 5:
        pass
    elif a < b < 10:
        pass

    # 4. 排序内置函数
    arr = [5, 2, 4, 1, 3]
    arr.sort()
    print("sorted array1: ", arr)
    arr = [5, 2, 4, 1, 3]
    arr2 = sorted(arr)
    print("sorted array2: ", arr2)



