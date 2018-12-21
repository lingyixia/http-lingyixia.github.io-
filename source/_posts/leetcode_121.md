---
layout: post
title: 股票最大利润
category: 算法
tags: [leetcode,动态规划]
description: 求股票最大利润
---

>Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

```
//第一种方法：记录到i-1为止，最小下标，然后比较prices[i]-prices[currintMinIndex]
int maxProfit(vector<int>& prices) 
{
	int currintMinIndex = 0;
	int currentMax = 0;
	int maxSum = 0;
	for(int i = 1;i<prices.size();i++)
	{
		if(prices[i-1]<prices[currintMinIndex])
		{
			currintMinIndex = i-1;
		}
		currentMax = prices[i] - prices[currintMinIndex];
		maxSum = maxSum>currentMax?maxSum:currentMax;
	}     
	return maxSum;
}
```
```
//第二种方法：计算prices[i]-price[i-1],然后题目就可以转为连续数组最大值问题
int maxProfit(vector<int>& prices) 
{
	int currentProfit = 0;
	int currentMaxProfit=0;
	int maxProfit = 0;//此处不能是INT_MIN，因为有可能[7,6,4,3,1]，即一直下降，此时可以选择不买不买，即利润最低为0
	for(int i = 1;i<prices.size();i++)
	{
		currentProfit = prices[i]-prices[i-1];
		currentMaxProfit += currentProfit;
		maxProfit = currentMaxProfit>maxProfit?currentMaxProfit:maxProfit;
		currentMaxProfit = currentMaxProfit<0?0:currentMaxProfit;
	}
	return maxProfit>0;
}
```