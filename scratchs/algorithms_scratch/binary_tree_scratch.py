# -*- coding: utf-8 -*-
"""
Created on 2024-09-04 13:11:31

@author: borisσ, Chairman of FrameX Inc.

I hope to use AI or LLMs to help people better understand the world and humanity.

We are big fans of xAI.

I am recently interested in Multimodal LLMs.
"""
from collections import deque


class TreeNode:
    def __init__(self, val:int):
        self.val: int = val
        self.left: TreeNode | None = None  # TreeNode类型或None类型
        self.right: TreeNode | None = None


class ArrayBinaryTree:
    def __init__(self, arr: list[int | None]):
        self._tree = list(arr)

    def size(self):
        return len(self._tree)

    def val(self, i: int) -> int | None:
        if i < 0 or i >= self.size():
            return None
        return self._tree[i]

    def left(self, i: int) -> int | None:
        return 2 * i + 1

    def right(self, i: int) -> int | None:
        return 2 * i + 2

    def parent(self, i: int) -> int | None:
        return (i - 1) // 2

    def level_order(self) -> list[int | None]:
        self.res = []
        for i in range(len(self._tree)):
            if self.val(i) is not None:
                self.res.append(self._tree[i])
        return self.res

    def dfs(self, i: int, order: str):
        if self.val(i) is None:
            return
        if order == 'pre':
            self.res.append(self.val(i))
        self.dfs(self.left(i), order)
        if order == 'in':
            self.res.append(self.val(i))
        self.dfs(self.right(i), order)
        if order == 'post':
            self.res.append(self.val(i))

    def pre_order(self) -> list[int | None]:
        self.res = []
        self.dfs(0, 'pre')
        return self.res

    def in_order(self) -> list[int | None]:
        self.res = []
        self.dfs(0, 'in')
        return self.res

    def post_order(self) -> list[int | None]:
        self.res = []
        self.dfs(0, 'post')
        return self.res


def left_order(root: TreeNode | None) -> list[int]:
    # BFS搜索算法:
    # 输入: 一个根节点, 输出: 树的val的层序遍历
    # 需要: 一个queue来按层顺序存储节点, 一个result列来保存结果
    # 时间空间复杂度均为: O(n)
    queue: deque[TreeNode] = deque()
    queue.append(root)
    result: list[int] = []
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        elif node.right:
            queue.append(node.right)
    return result


def pre_order(root: TreeNode | None) -> list[int]:
    # DFS搜索算法:
    # 输入: 一个根节点, 输出: 数的val的前中后序遍历
    # 需要: result数组空间
    # 时间空间复杂度均为: O(n)
    result: list[int] = []
    if root is None:
        return result
    result.append(root.val)
    pre_order(root.left)
    pre_order(root.right)
    return result


def in_order(root: TreeNode | None) -> list[int]:
    result: list[int] = []
    if root is None:
        return result
    in_order(root.left)
    result.append(root.val)
    in_order(root.right)
    return result


def post_order(root: TreeNode | None) -> list[int]:
    result: list[int] = []
    if root is None:
        return result
    post_order(root.left)
    post_order(root.right)
    result.append(root.val)
    return result


if __name__ == '__main__':

    # 1. 初始化节点
    n0 = TreeNode(0)
    n1 = TreeNode(1)
    n2 = TreeNode(2)
    n3 = TreeNode(3)
    n4 = TreeNode(4)
    n5 = TreeNode(5)

    # 2. 初始化引用
    n0.left = n1
    n0.right = n2
    n1.left = n3
    n1.right = n4
    n2.left = n5

    # 3. 增加节点p到0和1之间
    np = TreeNode(8)
    n0.left = np
    np.left = n1

    # 4. 删除0和1之间的节点p
    n0.left = n1

    # 5. 常见二叉树类型
    # (1) 满二叉树, perfect binary tree, 所有层被填满
    # (2) 完全二叉树, full binary tree, 仅底层可能未填满但节点均靠左
    # (3) 平衡二叉树, balanced binary tree, 左右子树的高度相差不超过1
    # 说明: 满二叉树为最理想树, 链表为最不理想树

    # 6. 层序遍历, level order traversal
    # 也算是广度优先搜索, breadth-first search, BFS
    BFS_test = left_order(n0)
    print(BFS_test)

    # 7. 表示任意二叉树
    # 二叉树的数组表示
    # 使用 None 来表示空位
    tree = [1, 2, 3, 4, None, 6, 7, 8, 9, None, None, 12, None, None, 15]
    tree = ArrayBinaryTree(tree)
    in_res = tree.in_order()
    print(in_res)
