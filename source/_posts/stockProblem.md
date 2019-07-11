---
title: 股票利润问题
date: 2019-06-22 18:50:47
category: 算法
tags: [leetcode,动态规划]
---
LeetCode中的股票问题
<!--more-->

#一次买卖
>>[Leetcode121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
)

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

#不限次数
>>[Leetcode122](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

```
 int maxProfit(vector<int>& prices) 
 {
        if(prices.empty()) return 0;
        int result=0;
        for(int i =1;i<prices.size();i++)
        {
            if(prices[i]>prices[i-1])
            {
                result+=prices[i]-prices[i-1];
            }
        }
        return result;
    }
```

#K次买卖
>>[Leetcode123](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)
动态规划:global[i][j]的意义是第i天以及之前最多进行j次交易所能获得的最大利润,local[i][j]的意义是在第i天以及之前最多进行j次交易且在第i天卖出股票所能得到的最大利润(如果global[i][j]是在第i天卖出,则global[i][j]=local[i][j]),地推公式为:
$$
\begin{align}
diff &= prices[i] - prices[i - 1] \\
local[i][j] &= max(global[i - 1][j - 1] + max(diff, 0), local[i - 1][j] + diff)\\
 global[i][j] &= max(global[i - 1][j], local[i][j]) \end{align}
$$
解释:
local公式:$global[i-1][j-1]+max(diff,0)$的意思是global[i-1][j-1]必然不可能在第i-1天买入(买入就没办法卖出了,利润自然要小),如果diff>0则在加上i-1天买,第i天卖的利润(一共j次交易)，否则就不算了(一共j-1次交易)。$local[i - 1][j] + diff$的意思是$第i-1天不卖了,改为第i天卖.
gobal公式较简单,不解释.

```
 int maxProfitK(int k, vector<int> &prices)
{
 if(prices.size()<2) return 0;       
    vector<vector<int>> local(prices.size(), vector<int>(k+1, 0));
    vector<vector<int>> global(prices.size(), vector<int>(k+1, 0));
    for (int i = 1; i < prices.size(); ++i)
    {
        int diff = prices[i] - prices[i - 1];
        for (int j = 1; j <= k; ++j)
        {
            local[i][j] = max(global[i - 1][j - 1] + max(diff, 0), local[i - 1][j] + diff);
            global[i][j] = max(global[i - 1][j], local[i][j]);
        }
    }
    return global[prices.size() - 1][k];
}
```

>>空间优化

```
int maxProfitK(int k, vector<int> &prices)
{
    if(prices.size()<2) return 0;
    vector<int> local(k + 1);
    vector<int> global(k + 1);
    vector<int> localCopy;
    vector<int> globalCopy;
    for (int i = 1; i < prices.size(); ++i)
    {
         localCopy = local;
        globalCopy = global;
        int diff = prices[i] - prices[i - 1];
        for (int j = 1; j <= k; ++j)
        {
            local[j] = max(globalCopy[j - 1] + max(diff, 0), localCopy[j] + diff);
            global[j] = max(global[j], local[j]);
        }
    }
    return global[k];
}
```
>>注意,当k>prices.size()/2的时候其实就退化成了**不限次数**问题，因此这里可以做一些优化.

#股票冷冻期
>>[Leetcode309](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)
动态规划:
buy[i]数组:第i天以及之前最后一个操作是buy所能得到的最大利润
sell[i]数组:第i天以及之前最后一个操作是sell所能得到的最大利润
cooldown[i]数组:第i天以及之前最后一个操作是cooldown所能得到的最大利润
地推公式为:
$$
\begin{align}
        buy[i] &= max(buy[i - 1], cooldown[i - 1] - prices[i]); \\
        sell[i] &= max(sell[i - 1], buy[i] + prices[i]); \\
        cooldown[i] &= max(cooldown[i - 1], max(buy[i - 1], sell[i - 1])); \\
\end{align}
$$
解释:
每一个公式的前一项都表示在第i天不进行相应的操作会怎样,第二项表示在第i天进行相应的操作会怎样.
buy:第一项表示如果在第i天不buy,则自然$buy[i]=buy[i-1]$,如果在第i天buy,则buy之前的状态必然是cooldown,因此$buy[i]=cooldown[i-1]-prices[i]$
sell:第一项表示如果在第i天不sell,则自然$sell[i]=sell[i-1]$,如果在第i天sell,则sell之前的状态必然是buy,因此$sell[i]=buy[i-1]+prices[i]$
cooldown:第一项表示如果在第i天cooldown,则自然$cooldown[i]=cooldown[i-1]$,如果在第i天cooldown,则cooldown[i]=max(sell[i-1],buy[i-1])

```
 int maxProfit(vector<int> &prices)
{
    if(prices.size()<2) return 0;
    vector<int> buy(prices.size());
    vector<int> sell(prices.size());
    vector<int> cooldown(prices.size());
    buy[0]=-prices[0];
    for (int i = 1; i < prices.size(); ++i)
    {
        buy[i] = max(buy[i - 1], cooldown[i - 1] - prices[i]);
        sell[i] = max(sell[i - 1], buy[i] + prices[i]);
        cooldown[i] = max(cooldown[i - 1], max(buy[i - 1], sell[i - 1]));
    }
    return sell.back();//利润最高的最后一次操作必然是sell
}
```

>>由于$cooldown[i] = sell[i-1]$(cooldown[i]的意思是i以及之前的最后一次操作是cooldown,也就是i-1以及之前的最后一次操作是sell,也就是sell[i-1])
地推公式可以归位两个:
$$
\begin{align}
buy[i]  &= max(sell[i-2] - price, buy[i-1]) \\
sell[i] &= max(buy[i-1] + price, sell[i-1])
\end{align}
$$

```
 int maxProfit(vector<int> &prices)
{
    if(prices.size()<2) return 0;
    vector<int> buy(prices.size());
    vector<int> sell(prices.size());
    buy[0]=-prices[0];
    sell[0]=0;
    buy[1]=-(prices[0]>prices[1]?prices[1]:prices[0]);
    sell[1]=prices[1]-prices[0]>0?prices[1]-prices[0]:0;
    for (int i = 2; i < prices.size(); ++i)
    {
        buy[i] = max(buy[i - 1], sell[i - 2] - prices[i]);
        sell[i] = max(sell[i - 1], buy[i] + prices[i]);
    }
    return sell.back();
```
>>最后优化

```
int maxProfit(vector<int>& prices) 
{
    int buy = INT_MIN, pre_buy = 0, sell = 0, pre_sell = 0;
    for (int price : prices) 
    {
        pre_buy = buy;
        buy = max(pre_sell - price, pre_buy);
        pre_sell = sell;
        sell = max(pre_buy + price, pre_sell);
    }
    return sell;
}
```