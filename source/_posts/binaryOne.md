---
title: 二进制中一的个数
date: 2019-02-14 17:11:03
category: 算法
tags: [leetcode,牛客网]
---
计算一个整数的二进制中1的个数
<!--more-->

1. 把1循环左移,同时和原数字做&操作,直到1变成0
```
int numberOfOne(int number)
{
	int count = 0;
	int flag = 1;
	while (flag)
	{
		if (number & flag) count++;
		flag <<= 1;
	}
	return count;
}
```

2. 一个整数number&(number)结果是将number最右侧的1变为0
```
int numberOfOne(int number)
{
	int count=0;
	while (number)
	{
		number &= (number - 1);
		count++;
	}
	return count;
}
```