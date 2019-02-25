---
title: 多叉树最大深度
date: 2019-01-19 17:24:25
category: 算法
sage: true
hidden: true
tags: [tree,遍历,leetcode]
---
>题意为输出多叉树最大深度,最容易想到的方式是递归,该题记录重点为第二种非递归方式,其实也相当于多叉树的**层序遍历**。

#递归
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

#非递归
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