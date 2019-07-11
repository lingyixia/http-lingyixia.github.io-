---
layout: post
title: PatchArray
date: 2019-01-14 21:52:31
category: 算法 
tags: [leetcode,贪心算法]
---
>Given a sorted positive integer array nums and an integer n, add/patch elements to the array such that any number in range [1, n] inclusive can be formed by the sum of some elements in the array. Return the minimum number of patches required.

举例:nums = [1, 2, 4, 13, 43],n = 100,当已经将[1,2,4]加入临时数组时,此时有子数组[1],[2],[1,2],[4],[1,4],[2,4],[1,2,4],即连续的1,2,3,4,5,6,7,最大是7,下一个加入临时数组的数字需要将8加进去,但是nums中下一个是13,就算加进去也没有8(中间有裂口),那么**要想中间没有裂口且数组和增加最大,下一个加入临时数组的数字是就得是8(贪心)**,此时子数组最大和是7+8=15,而下一个miss就是16。

实例代码:
```
int minPatches(vector<int>& nums, int n)
{
	long long miss = 1;
	int res = 0;
	int pos = 0;
	while (miss<=n)
	{
		if (pos<nums.size()&&nums[pos]<=miss)
		{
			miss += nums[pos++];
		}
		else
		{
			miss += miss;
			res++;
		}
	}
	return res;
}
```