# algorithms_scratch

Summarize Leetcode solutions from scratch.

## week_00

|problems|tags|tricks|
|:----:|:-----:|:-----:|
|[701. 二叉搜索树中的插入操作](https://leetcode.cn/problems/insert-into-a-binary-search-tree/description/)|BST, Recursion|首先判断空node的情况, 要额外定义函数. val和root.val比大小, 决定往左往右走, 随后进行Recursion, 最后填None的空位|
|[98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/description/)|BST, Recursion, 分治法|要额外定义函数. 空node返回True, 满足minval<node.val<maxval, 随后递归, 对node输入负无穷到正无穷判定|
|[104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/)|Recursion, 分治法|首先判断空node的情况, 随后分治法就完了.|
|[103. 二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/description/)|BFS|需要定义仨: 结果的[[], [], []], 装一层节点的collections.deque, start_from_left的bool. **技巧**: 1. 从左往右的时候, popleft, 左右左右地append, 2. 从右往左地时候, pop, 右左右左地appendleft.|
