---
title: 数组中数量超过一半
date: 2019-03-13 14:53:47
category: [算法]
tags: 
---
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
<!--more-->

```
int MoreThanHalfNum(vector<int> numbers)
{
	int result = numbers[0];//擂主一开始是第一个数字 
	int time = 1;//分数是1
	for (int i = 1; i < numbers.size(); i++)
	{
		if (time == 0)//如果分数掉到0了
		{
			result = numbers[i];//有新的挑战者上台直接当擂主
			time=1;
		}
		else if (numbers[i] == result) time++; // 如果和擂主同族(数字一样)分数+1,相当于给擂主加HP 
		else time--;//如果不同族(数字不一样) //掉一分,相当于擂主花1HP打掉1HP的挑战者 
	}
	int resultNum = 0;
	for (int i = 0; i < numbers.size(); i++)
	{
		if (numbers[i] == result) resultNum++;
	}
	return resultNum>=numbers.size()/2?resultNum:0;
}
```
