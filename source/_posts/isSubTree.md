---
title: isSubTree
date: 2019-03-01 18:56:15
category: 算法
tags: [树]
sage: true
hidden: true
---
>输入两棵二叉树A，B，判断B是不是A的子结构。(ps：我们约定空树不是任意一个树的子结构)

```
 bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        if(!pRoot1 || !pRoot2) return false;
        bool result = false;
        if(pRoot1->val==pRoot2->val) result = isSubTree(pRoot1,pRoot2);
        if(!result)  result = HasSubtree(pRoot1->left,pRoot2);
        if(!result)  result = HasSubtree(pRoot1->right,pRoot2);
        return result;
    }
    bool isSubTree(TreeNode* pRoot1,TreeNode* pRoot2)
    {
        if(!pRoot2) return true;
        if(!pRoot1) return false; 
        bool result = false;
        if(pRoot1->val==pRoot2->val) result = true;
        if(result) result = isSubTree(pRoot1->left,pRoot2->left);
        if(result) result = isSubTree(pRoot1->right,pRoot2->right);
         return result;
    }
```
需要注意得地方是返回bool判断左树右树的逻辑
