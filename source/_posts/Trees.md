---
title: 树的常见算法
date: 2019-07-11 12:20:03
category: 算法
tags: [Tree]
---

#二叉树的遍历
##preOrderTraversal

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

```
#先序简单
vector<int> preorderTraversal(TreeNode* root) 
{
	vector<int> result;
    if(!root) return result;
    stack<TreeNode*> s;
    TreeNode* cursor = root;
    s.push(root);
    while(!s.empty())
    {
        cursor=s.top();
        result.push_back(cursor->val);
        s.pop();
        if(cursor->right)
        {
            s.push(cursor->right);
        }
        if(cursor->left)
        {
            s.push(cursor->left);
        }
    }
    return result;
}
```

##inOrderTraversal
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

##postOrderTraversal
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

```
#后序拿offer算法
struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 };
 vector<int> postorderTraversal(TreeNode* root) 
 {
    stack<TreeNode*> s;
    stack<TreeNode*> resultTemp;
    vector<int> result;
    if(!root) return result;
    TreeNode* cursor = root;
    s.push(cursor);
    while(!s.empty())
    {
        cursor=s.top();
        s.pop();
        resultTemp.push(cursor);
        if(cursor->left) s.push(cursor->left);
        if(cursor->right) s.push(cursor->right);
    }
    while(!resultTemp.empty())
    {
        result.push_back(resultTemp.top()->val);
        resultTemp.pop();
    }
    return result;
}
```
##多叉树层序遍历
>>这个和多叉树的非递归深度重复

```
class Node {
public:
    int val;
    vector<Node*> children;
    Node() {}
    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};
class Solution
{
public:
    vector <vector<int>> levelOrder(Node *root)
    {
        vector <vector<int>> result;
        if (!root)
        {
            return result;
        }
        queue < Node * > q;
        q.push(root);
        Node *cursor = root;
        vector<int> temp;
        while (!q.empty())
        {
            temp.clear();
            int size = q.size();
            for (int i = 0; i < size; i++)
            {
                cursor = q.front();
                temp.push_back(cursor->val);
                q.pop();
                for (int j = 0; j < cursor->children.size(); j++)
                {
                    q.push(cursor->children[j]);
                }
            }
            result.push_back(temp);
        }
        return result;
    }
};
```
>>如果是二叉树只需要两个循环就可以了，因为最内层循环只需要改成左右子树即可

#判断子树
>>输入两棵二叉树A，B，判断B是不是A的子结构。(ps：我们约定空树不是任意一个树的子结构)

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

#多叉树最大深度
>>题意为输出多叉树最大深度,最容易想到的方式是递归,该题记录重点为第二种非递归方式,其实也相当于多叉树的**层序遍历**。

##递归
```
#include<iostream>
#include<queue>
#include<vector>
using namespace std;
class Node {
public:
	int val;
	vector<Node*> children;
	Node() {}
	Node(int _val, vector<Node*> _children) {
		val = _val;
		children = _children;
	}
};
int maxDepth(Node* root);
int main()
{
	int maxDepth(Node* root);
	return 0;
}
int maxDepth(Node* root) 
{
    if(!root)//出口条件
    {
        return 0;
    }
    int maxDep=0;
    for(int i =0;i<root->children.size();i++)
    {
        maxDep=max(maxDep,maxDepth(root->children[i]));
    }
    return maxDep+1;
}
```

##非递归
```
#include<iostream>
#include<queue>
#include<vector>
using namespace std;
class Node {
public:
	int val;
	vector<Node*> children;
	Node() {}
	Node(int _val, vector<Node*> _children) {
		val = _val;
		children = _children;
	}
};
int maxDepth(Node* root);
int main()
{
	int maxDepth(Node* root);
	return 0;
}
int maxDepth(Node* root)
{
    if(!root)//只是当root为空时候的一个判断
    {
        return 0;
    }
	int depth = 0;
	queue<Node*> q;
	q.push(root);
	Node* cursor = NULL;
	while (!q.empty())//这里的三层循环第二层目的是将每一层都组成一个数组,并计算层序,如果单纯为了输出层序遍历的结果完全可以使用两层循环
	{
	    int size = q.size();
	    for (int i = 0; i < size;i++)//注意该队列其实是一层一层从做到右入队列的,切不可用while(!q.empty())
	    {
		    cursor = q.front();
		    q.pop();
		    for (int j = 0; j < cursor->children.size(); j++)
		    {
			    q.push(cursor->children[j]);
		    }
		}
                depth++;//每层加1
	}
    return depth;
}
```
>>注意，两个方法中的`if(!root)`作用并不一样，对于递归而言是跳出条件，而对于非递归而言只是一种情况

#二叉树镜像
>>把一个二叉树变为镜像

```
void Mirror(TreeNode *pRoot) 
{
    if(pRoot)
    {
        TreeNode* node = pRoot->left;
        pRoot->left=pRoot->right;
        pRoot->right=node;
        Mirror(pRoot->left);
        Mirror(pRoot->right);
    }
}
```
#对称二叉树

```
bool judge(TreeNode* p,TreeNode* q)
{
    if(!p && !q) return true;
    if(!p || !q) return false;
    if(p->val != q->val) return false;
    return judge(p->left,q->right) && judge(p->right,q->left);
}
bool isSymmetrical(TreeNode* pRoot)
{
    return judge(pRoot,pRoot);
}
```


#树的子结构(或子树)
>>给定两棵二叉树A和B，判断A是不是B的子结构或子树，子结构要求A是B的一部分即可，但是子树要求叶子结点也得相同.

```
bool judgeSubTree(TreeNode *pRoot1, TreeNode *pRoot2)
{
    //判断子结构
    if (!pRoot2) return true;
    if (!pRoot1) return false;
    //判断子树
    //if (!pRoot1 && !pRoot2) return true;
    //if (!(pRoot1 && pRoot2)) return false;
    bool judge = pRoot1->val == pRoot2->val;
    if (judge) judge = judgeSubTree(pRoot1->left, pRoot2->left);
    if (judge) judge = judgeSubTree(pRoot1->right, pRoot2->right);
    return judge;
}

bool isSubtree(TreeNode *pRoot1, TreeNode *pRoot2)
{
    if (!pRoot1 || !pRoot2) return false;
    bool judge = judgeSubTree(pRoot1, pRoot2);
    if (!judge) judge = isSubtree(pRoot1->left, pRoot2);
    if (!judge) judge = isSubtree(pRoot1->right, pRoot2);
    return judge;
}
```

#二叉树中路径和
>>输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

```
vector<vector<int>> result;
void FindPath(TreeNode* root,int expectNumber,vector<int>& path)
{
    if(!root) return;
    path.push_back(root->val);
    expectNumber-=root->val;
    if(!root->left && !root->right && !expectNumber)
    {
        result.push_back(path);
    }
    if(root->left)
    {
        FindPath(root->left,expectNumber,path);   
    }
    if(root->right)
    {
        FindPath(root->right,expectNumber,path);   
    }
    path.pop_back();
}
vector<vector<int> > FindPath(TreeNode* root,int expectNumber) 
{
    if(!root) return result;
    vector<int> path;
    FindPath(root,expectNumber,path);
    return result;
}
```
#[二叉树所有路径中和最大](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/
>>这里的路径指的是树上所有能联通的两个节点之间的路径

```
int maxSumResult = 0;

int maxPathSum(TreeNode *root)
{
    if (!root)
        return 0;
    int left = maxPathSum(root->left);
    int right = maxPathSum(root->right);
    int temp = root->val;
    if (left > 0)
    {
        temp += left;
    }
    if (right > 0)
    {
        temp += right;
    }
    maxSumResult = max(maxSumResult, temp);
    return max(root->val, max(root->val + left, root->val + right));
}
```
>>调用maxPathSum后maxSumResult得到的就是最大路径和

#判断是不是平衡二叉树

```
int getHeight(TreeNode* root,bool& ifBalance)
{
    if(!root) return 0;
    int left = getHeight(root->left,ifBalance);
    int right =getHeight(root->right,ifBalance);
    if(abs(left-right)>1)
    {
        ifBalance=false;
    }
    return left>right?left+1:right+1;
}
bool IsBalanced_Solution(TreeNode* pRoot) 
{
    bool ifBalance=true;
    getHeight(pRoot,ifBalance);
    return ifBalance;
}
```
#next指针
>>完全二叉树增加next指针指向右侧节点

>>只适用于完全二叉树
```
Node *connectCore(Node *root)
{
    if (!root->left && !root->right) return NULL;
    root->left->next = root->right;
    connectCore(root->left);
    if (root->next)
    {
        root->right->next = root->next->left;
    }
    connectCore(root->right);
    return root;
}

Node *connect(Node *root)
{
    if (!root || (!root->left && !root->right))
        return root;
    return connectCore(root);
}
```

>>层序遍历，适用于任何形式的二叉树

```
Node *connect(Node *root)
{
    if(!root ||(!root->left && !root->right)) return root;
    Node *cursor = root;
    queue<Node *> q;
    q.push(cursor);
    while (!q.empty())
    {
        int size = q.size();
        for (int i = 0; i < size; ++i)
        {
            cursor = q.front();
            q.pop();
            cursor->next = (i==size-1)? nullptr:q.front();
            if(cursor->left)
            {
                q.push(cursor->left);
            }
            if(cursor->right)
            {
                q.push(cursor->right);
            } 
        }
    }
    return root;
```