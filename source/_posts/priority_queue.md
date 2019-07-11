---
layout: post
title: C++优先队列
category: 基础
date: 2019-1-11 19:11:11
tags: [C++]
description: C++优先队列简单简单介绍
---

优先队列是队列和栈结合形成的一种数据结构，c++中使用堆来构建。
<!--more-->
首先函数在头文件<queue>中，归属于命名空间std，使用的时候需要注意。
有两种声明方式:
```
std::priority_queue<T> pq;
std::priority_queue<T, std::vector<T>, cmp> pq;
```
第一种方式较为简单，当T是基本数据类型的时候使用
当T是自定义数据结构的时候(一般是struct)需要使用第二中声明方式.三个参数从左到右第一个指的是优先队列中的数据类型，第二个表示存储堆的数据结构，一般是vector即可，第三个是比较结构，是一个struct，默认是less，即小的在前，但是默认声明方式只针对基本数据结构，自定义的需要重写该结构体，less也可以改为greater。
>>greater和less是std实现的两个仿函数（就是使一个类的使用看上去像一个函数。其实现就是类中实现一个operator()，这个类就有了类似函数的行为，就是一个仿函数类了）

eg:  
```
#include<iostream>
#include<queue>
using namespace std;
struct Node
{
	int val;
	Node(int val) :val(val) {}

};
int main()
{
	priority_queue<Node> A;                    //大根堆
	priority_queue<Node, vector<Node>, greater<Node> > B;    //小根堆 
	A.push(Node(1));
	A.push(Node(9));
	A.push(Node(4));
	cout << A.top().val << endl;
	A.pop();
	cout << A.top().val << endl;
	A.pop();
	cout << A.top().val << endl;
	A.pop();
	return 0;
}
```
这样运行出错，因为priority_queue<Node> A; 默认使用的是less，而less的源代码是:

```
struct less
{	// functor for operator<
	_CXX17_DEPRECATE_ADAPTOR_TYPEDEFS typedef _Ty first_argument_type;
	_CXX17_DEPRECATE_ADAPTOR_TYPEDEFS typedef _Ty second_argument_type;
	_CXX17_DEPRECATE_ADAPTOR_TYPEDEFS typedef bool result_type;

	constexpr bool operator()(const _Ty&_Left, const _Ty& _Right) const
	{	// apply operator< to operands
		return (_Left < _Right);
	}
};
```

>>return (_Left < _Right)对于一个自定义结构体而言程序不知道<是什么操作，因此需要要么在结构体中重载<符号，要么重写less函数，因此有两种写法。  
第一种: 

```
struct Node
{
	int val;
	Node(int val) :val(val) {}
	bool operator <(Node a) const { return val < a.val; }
	bool operator >(Node a) const { return val > a.val; }
};
```  

第二种:  

```
#include<iostream>
#include<queue>
using namespace std;
struct Node
{
	int val;
	Node(int val) :val(val) {}

};
struct cmp
{
	bool operator()(Node a, Node b)
	{
		return a.val < b.val;
	}
};
int main()
{
	priority_queue<Node,vector<Node>,cmp> A;                    //大根堆
	priority_queue<Node, vector<Node>, less<Node> > B;    //小根堆 
	A.push(Node(1));
	A.push(Node(9));
	A.push(Node(4));
	cout << A.top().val << endl;
	A.pop();
	cout << A.top().val << endl;
	A.pop();
	cout << A.top().val << endl;
	A.pop();
	return 0;
}
```