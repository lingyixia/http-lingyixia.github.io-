---
title: BinaryTreeTraversal
date: 2019-01-19 11:01:04
category: 算法
tags: [tree,遍历]
---
>二叉树的先序、中序、后序非递归遍历

#preOrderTraversal

```
struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 };
vector<int> preorderTraversal(TreeNode* root)
{
	vector<int> result;
	stack<TreeNode*> treeNodeStack;
	TreeNode* cursor = root;
	while (cursor || treeNodeStack.size()>0)
	{
		while (cursor)
		{
			result.push_back(cursor->val);
			treeNodeStack.push(cursor);
			cursor = cursor->left;
		}
		cursor = treeNodeStack.top();
		treeNodeStack.pop();
		cursor = cursor->right;
	}
	return result;
}
```

#inOrderTraversal
```
struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 };
vector<int> inorderTraversal(TreeNode* root)
{
	vector<int> result;
	stack<TreeNode*> treeNodeStack;
	TreeNode* cursor = root;
	while (cursor || treeNodeStack.size()>0)
	{
		while (cursor)
		{
			treeNodeStack.push(cursor);
			cursor = cursor->left;
		}
		cursor = treeNodeStack.top();
		result.push_back(cursor->val);
		treeNodeStack.pop();
		cursor = cursor->right;
	}
	return result;
}
```
>>只是简单的把先序遍历28行挪了一下

```
vector<int> postorderTraversal(TreeNode* root)
{
	vector<int> result;
      if(!root)
      {
         return result;
      }
	stack<TreeNode*> treeNodeStack;
	TreeNode* preNode = NULL;
	TreeNode* cursor = root;
	treeNodeStack.push(cursor);
	while (treeNodeStack.size() > 0)
	{
		cursor = treeNodeStack.top();
		if ((!cursor->left && !cursor->right) || (preNode && preNode == cursor->left) ||  (preNode && preNode == cursor->right))
		{
			result.push_back(cursor->val);
			preNode = cursor;
			treeNodeStack.pop();
		}
		else
		{
			if (cursor->right)
			{
				treeNodeStack.push(cursor->right);
			}
			if (cursor->left)
            {
				treeNodeStack.push(cursor->left);
			}
		}
	}
	return result;
}
```
#postOrderTraversal
```
struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 };
vector<int> postorderTraversal(TreeNode* root)
{
	vector<int> result;
      if(!root)
      {
         return result;
      }
	stack<TreeNode*> treeNodeStack;
	TreeNode* preNode = NULL;
	TreeNode* cursor = root;
	treeNodeStack.push(cursor);
	while (treeNodeStack.size() > 0)
	{
		cursor = treeNodeStack.top();
		if ((!cursor->left && !cursor->right) || (preNode && preNode == cursor->left) ||  (preNode && preNode == cursor->right))
		{
			result.push_back(cursor->val);
			preNode = cursor;
			treeNodeStack.pop();
		}
		else
		{
			if (cursor->right)
			{
				treeNodeStack.push(cursor->right);
			}
			if (cursor->left)
            {
				treeNodeStack.push(cursor->left);
			}
		}
	}
	return result;
}
```