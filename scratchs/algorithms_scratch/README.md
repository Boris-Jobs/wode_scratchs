# algorithms_scratch

Summarize Leetcode solutions from scratch.

## week_00
主要刷了树的内容.

|problems|tags|tricks|
|:----:|:-----:|:-----:|
|[701. 二叉搜索树中的插入操作](https://leetcode.cn/problems/insert-into-a-binary-search-tree/description/)|BST, Recursion|首先判断空node的情况, 要额外定义函数. val和root.val比大小, 决定往左往右走, 随后进行Recursion, 最后填None的空位|
|[98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/description/)|BST, Recursion, 分治法|要额外定义函数. 空node返回True, 满足minval<node.val<maxval, 随后递归, 对node输入负无穷到正无穷判定|
|[104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/)|Recursion, 分治法|首先判断空node的情况, 随后分治法就完了.|
|[103. 二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/description/)|BFS|需要定义仨: 结果的[[], [], []], 装一层节点的collections.deque, start_from_left的bool. **技巧**: 1. 从左往右的时候, popleft, 左右左右地append, 2. 从右往左地时候, pop, 右左右左地appendleft.|
|[107. 二叉树的层序遍历 II](https://leetcode.cn/problems/binary-tree-level-order-traversal-ii/description/)|BFS|return res[::-1]|
|[102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/description/)|BFS|需要定义: 1. 存当前层的collections.deque, 2. 这个deque的len, 3. 存结果的res = [], 每个层循环append一个[]. **技巧**: popleft, 左右左右地append.|
|[236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/)|DFS|从上往下寻找祖先, 先判断: 左树右数是否都有返回祖先, 有就返回root; 若左有右没有, 那么左就是祖先; 若右有左没有, 那么右节点是祖先, 因为右包含了可能含有另一个节点的分支. |
|[110. 平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/description/)|DFS, 分治法|分别找左节点和右节点的最大深度, 返回当前节点以下的最大深度和当前节点左树有树差距是否小于2的布尔值|
|[124. 二叉树中的最大路径和 (困难)](https://leetcode.cn/problems/binary-tree-maximum-path-sum/description/)|DFS|用'左+root+右'来更新全局的path变量, 用'左+root'或'右+root'来返回单分支的值.|