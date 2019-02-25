---
title: Jump Game
date: 2019-01-15 21:03:15
category: 算法
sage: true
hidden: true
tags: [贪心算法,动态规划]
---

>Given an array of non-negative integers, you are initially positioned at the first index of the array.
Each element in the array represents your maximum jump length at that position.
Determine if you are able to reach the last index.
(可以不是正好最后一个坐标,能到最后就行)

贪心一:
```
bool canJump(vector<int>& nums) 
{
	int reach = 0;
	for (int i =0; i < nums.size(); i++)
	{
		if (i > reach || i >= nums.size() - 1) break;
		else
		{
			reach = max(reach, i + nums[i]);
		}
	}
	return reach >= nums.size() - 1;
}
```

>>每到一个i,如果i<=reach意味着[0,i-1]的坐标能达到reach,如果i>reach,则意味着根本就到不了这里,无需继续。

贪心二:
```
bool canJump(vector<int>& nums) 
{
	int len = nums.size();
	int curMax = nums[0];

	for (int i = 0; i <= curMax; i++)
	{
		if (nums[i] + i >= len - 1) return true;
		curMax = max(curMax, nums[i] + i);
	}
	return false;
}
```
>>两种形式不一样,思想差不多

动态规划:
```
bool canJump(vector<int>& nums) 
{
	vector<int> dp(nums.size(), 0);
	for (int i = 1; i < nums.size(); ++i) 
	{
		dp[i] = max(dp[i - 1], nums[i - 1]) - 1;
		if (dp[i] < 0) return false;
	}
	return true;
}
```
>>dp数组维护到位置i时剩余步骤的最大值.递推公式为:$dp[i] = max(dp[i - 1], nums[i - 1]) - 1$
