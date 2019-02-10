---
layout: post
title: 连续数组最大和
category: 算法
tags: [leetcode,动态规划]
description: 求连续数组最大和
---

>Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

```
//动态规划
//递推公式为DP[i] = max{DP[i-1] + A[i],DP[i-1]}
int maxSubArray(vector<int>& nums)
{
	int currentMax = 0;//currentMax是i处以及之前的连续最大值
	int max = INT_MIN;
	for (int i = 0; i < nums.size(); i++)
	{
		currentMax += nums[i];
		max = currentMax>max?currentMax:max;
		currentMax = currentMax<0?0:currentMax;
		//if (currentMax>max)
		//{
		//	max = currentMax;
		//}
		//if (currentMax<0)//之所以不用else if是考虑譬如[-2,1]的情况
		//{
		//	currentMax = 0;
		//}
	}
	return max;
}
```