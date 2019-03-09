---
layout: post
title: 连续数组最大和
category: 算法
tags: [leetcode,动态规划]
date: 2019-1-10 10:11:08
description: 求连续数组最大和
---

>Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

```
//动态规划
//递推公式为DP[i] = max{DP[i-1] + A[i],DP[i]},对于当前点i,要么与前面连起来组成和,要么自己组成和
int maxSubArray(vector<int>& nums) 
{
	int before = 0;//before就是Dp[i-1]
	int maxSum = INT_MIN;
	for (int i = 0; i <nums.size(); i++)
	{
		before = max(before+nums[i],nums[i]);
		maxSum = max(maxSum,before);
	}
	return maxSum;
}
```