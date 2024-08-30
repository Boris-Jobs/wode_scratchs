# -*- coding: utf-8 -*-
"""
Created on 2024-08-30 11:48:42

@author: borisσ, Chairman of FrameX Inc.

I hope to use AI or LLMs to help people better understand the world and humanity.

We are big fans of xAI.

I am recently interested in Multimodal LLMs.
"""

class TreeNode:
    """
    二叉树分类: 满二叉树, 完全二叉树, 二叉搜索树(有序树), 平衡二叉搜索树(左右高度差≤1)

    二叉树存储方式: 顺序存储(左子树下标2*i+1, 右子树下标2*i+2), 链式存储

    二叉树遍历方式: 深度优先遍历(前序/中序/后序遍历, 递归法/迭代法, 前中后指的是中间节点的遍历顺序), 广度优先遍历(层序遍历, 迭代法)
    1. 深度优先遍历--用栈实现
    2. 广度优先遍历--用队列实现

    """
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def preorderTraversal(self, root):
        res = []
        def preorder(node):
            if node is None:
                return
            res.append(node.val)
            preorder(node.left)
            preorder(node.right)
        preorder(root)
        return res


if __name__ == '__main__':
    root = TreeNode(0)
    root.left = TreeNode(1)
    root.right = TreeNode(2)
    root.left.left = TreeNode(3)
    root.left.right = TreeNode(4)
    root.right.left = TreeNode(5)
    root.right.right = TreeNode(6)
    print("The result of preorderTraversal is ", root.preorderTraversal(root), ".")
