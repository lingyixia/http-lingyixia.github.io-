---
title: 贪心
date: 2019-07-27 10:37:31
category: 算法
tags:
---

#[Jump Game](https://leetcode.com/problems/jump-game/)
贪心一:
>>每到一个i,如果i<=reach意味着[0,i-1]的坐标能达到reach,如果i>reach,则意味着根本就到不了这里,无需继续。

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
贪心二:
>>和上诉形式不一样,思想差不多

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

动态规划:
>>dp[i]表示到达i时候最多还剩下多少步

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
#推销员
>>阿明是一名推销员，他奉命到螺丝街推销他们公司的产品。螺丝街是一条死胡同，出口与入口是同一个，街道的一侧是围墙，另一侧是住户。螺丝街一共有 N 家住户，第 i 家住户到入口的距离为 Si 米。由于同一栋房子里可以有多家住户，所以可能有多家住户与入口的距离相等。阿明会从入口进入，依次向螺丝街的 X 家住户推销产品，然后再原路走出去。
阿明每走 1 米就会积累 1 点疲劳值，向第 i 家住户推销产品会积累 Ai 点疲劳值。阿明是工作狂，他想知道，对于不同的 X，在不走多余的路的前提下，他最多可以积累多少点疲劳值。

暂存

#
