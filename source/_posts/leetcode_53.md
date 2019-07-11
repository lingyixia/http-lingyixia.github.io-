---
layout: post
title: 连续数组最大和
category: 算法
tags: [leetcode,动态规划]
date: 2019-1-10 10:11:08
description: 求连续数组最大和
---

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
<!--more-->

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
>>矩阵最大子矩阵和

```
int calclineSum(vector<int> array)//完全和上面一样
{
	int before=0;
	int maxSum = INT_MIN;
	for (int i = 0; i < array.size(); i++)
	{
		before = max(before+array[i],array[i]);
		maxSum = max(maxSum,before);
	}
	return maxSum;
}

int maxSumMatrix(vector<vector<int>> matrix)
{
	int maxSum = INT_MIN;
	for (int i = 0; i < matrix.size(); i++)
	{
		vector<int> lineSum(matrix[0].size());
		for (int j = i; j < matrix.size(); j++)
		{
			for (int k = 0; k < matrix[0].size(); k++)
			{
				lineSum[k] += matrix[j][k];
			}
			maxSum = max(maxSum, calclineSum(lineSum));
		}
	}
	return maxSum;
}
```