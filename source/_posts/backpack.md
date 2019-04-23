---
layout: post
title: 背包问题
category: 算法
date: 2019-01-20 19:11:08
tags: [递归,背包,动态规划]
description: 三种背包问题
---

>0-1背包、完全背包、多重背包问题简记.一般而言,背包问题有两种:**要求恰好装满**和**不要求恰好装满**,其实两者**仅仅**初始化数组不同,前者要初始化`values[i][0]=0`,其他全为`-INT_MIN`.后者初始化全为0即可。

#0-1背包问题
##常规写法
递推公式为:`values[i][j] = max(values[i - 1][j], values[i-1][j - woods[i-1].volume] + woods[i-1].value);`
```
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
struct Woods
{
	int volume;
	int value;
};
void backPack(vector<Woods>& woods, int bag);
void findItems(vector<vector<int>>& values, vector<Woods>& woods, vector<int>& items, int i, int j);
int main()
{
	vector<Woods> woods = { Woods{2,3},Woods{3,4},Woods{4,5},Woods{5,6} };
	backPack(woods, 8);
	return 0;
}
void backPack(vector<Woods>& woods, int bag)
{
	vector<vector<int>> values(woods.size() + 1, vector<int>(bag + 1));
	for (int i = 1; i <= woods.size(); i++)
	{
		for (int j = 1; j <= bag; j++)
		{
			if (j < woods[i - 1].volume)//装不进去
			{
				values[i][j] = values[i - 1][j];
			}
			else//能装进去
			{
				values[i][j] = max(values[i - 1][j], values[i - 1][j - woods[i - 1].volume] + woods[i - 1].value);
			}
		}
	}
	vector<int> items(woods.size() + 1);//保存的是最终放入了哪些物品
	findItems(values, woods, items, woods.size(), bag);
}
void findItems(vector<vector<int>>& values, vector<Woods>& woods, vector<int>& items, int i, int j)
{
	if (i > 0)
	{
		if (values[i][j] == values[i - 1][j])
		{
			items[i] = 0;
			findItems(values, woods, items, i - 1, j);
		}
		else
		{
			items[i] = 1;
			findItems(values, woods, items, i - 1, j - woods[i - 1].volume);
		}
	}
}
```
上诉方法使用二维数组保存中间值values,values[i][j]表示当背包空间为j,允许使用的物品只有前两个的时候的最大价值，比较消耗空间，而且我们可以看出，对于每一步要求的`values[i][j]`而言，只依赖于`values[i-1][:]`,即前一行，因此我们只需要一维数组即可，循环保存前一行数据。但是这样貌似就无法找物品组成了，回头在想。
##空间优化写法
```
void backPack(vector<Woods>& woods, int bag)
{
	vector<int> values(bag + 1);
	for (int i = 1; i <= woods.size(); i++)
	{
		for (int j = bag; j >=1; j--)
		{
			if (j >= woods[i - 1].volume)//放不进去
			{
				values[j] = max(values[j], values[j - woods[i - 1].volume] + woods[i - 1].value);
			}
		}
	}
}
```
##再次改善
```
void backPack(vector<Woods>& woods, int bag)
{
	vector<int> values(bag + 1);
	for (int i = 1; i <= woods.size(); i++)
	{
		for (int j = bag; j >= woods[i - 1].volume; j--)//倒序
		{
			values[j] = max(values[j], values[j - woods[i - 1].volume] + woods[i - 1].value);
		}
	}
}
```

#完全背包问题
##常规写法
与0-1背包唯一的不同就是递推公式：`values[i][j] = max(values[i - 1][j], values[i][j - woods[i-1].volume] + woods[i-1].value);`不在详细说明

##空间优化写法
```
void backPack(vector<Woods>& woods, int bag)
{
	vector<int> values(bag + 1);
	for (int i = 1; i <= woods.size(); i++)
	{
		for (int j = woods[i - 1].volume; j <=bag ; j--)//正序
		{
			values[j] = max(values[j], values[j - woods[i - 1].volume] + woods[i - 1].value);
		}
	}
}

```

#多重背包问题（即每种物品的数量有上限）
##常规写法
在0-1背包的基础上增加数量循环判断,递推公式为:`values[i][j] = max(values[i][j], values[i-1][j - k * woods[i - 1].volume] + k * woods[i - 1].value);`
```
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
struct Woods
{
	int volume;
	int value;
	int num;
};
void backPack(vector<Woods>& woods, int bag);
int main()
{
	vector<Woods> woods = { Woods{ 80 ,20 ,4 },Woods{ 40 ,50, 9 },Woods{ 30 ,50, 7 },Woods{ 40 ,30 ,6 },Woods{ 20 ,20 ,1 } };
	backPack(woods, 1000);
	return 0;
}
void backPack(vector<Woods>& woods, int bag)
{
	vector<vector<int>> values(woods.size() + 1, vector<int>(bag + 1));
	for (int i = 1; i <= woods.size(); i++)
	{
		for (int j = 1; j <= bag; j++)
		{
			for (int k = 1; k <= woods[i - 1].num; k++)
			{
				if (j < k * woods[i - 1].volume) break;
				values[i][j] = max(values[i-1][j], values[i - 1][j - k * woods[i - 1].volume] + k * woods[i - 1].value);
			}
		}
	}
}
```
##空间优化写法
```
void backPack(vector<Woods>& woods, int bag)
{
	vector<int> values(bag + 1);
	for (int i = 1; i <= woods.size(); i++)
	{
		for (int j = bag; j >= 0; j--)
		{
			for (int k = 1; k <= woods[i - 1].num; k++)
			{
				if (j < k * woods[i - 1].volume) break;
				values[j] = max(values[j], values[j - k * woods[i - 1].volume] + k * woods[i - 1].value);
			}
		}
	}
}
```
#变种
##硬币问题1
leetcode322(最少硬币数量)
```
int coinChange(vector<int>& coins, int amount) {
	vector<vector<int>> values(coins.size() + 1, vector<int>(amount + 1,INT_MAX-1));
	for (int i = 1; i <= coins.size(); i++)
	{
		values[i][0] = 0;
		for (int j = 1; j <= amount; j++)
		{
			if (j < coins[i - 1])
			{
				values[i][j] = values[i - 1][j];
			}
			else
			{
				values[i][j] = min(values[i - 1][j], values[i][j - coins[i - 1]] + 1);
			}
		}
	}
	return values[coins.size()][amount]==INT_MAX-1?-1:values[coins.size()][amount];
}
```
>>values[i][j]表示使用前i种硬币组成j至少需要多少个硬币

简化空间:

```
int coinChange(vector<int>& coins, int amount) 
    {
	    vector<int> values(amount + 1,INT_MAX-1);
        values[0]=0;
	    for (int i = 1; i <= coins.size(); i++)
	    {
		    for (int j = coins[i - 1]; j <= amount; j++)
		    {
			    values[j] = min(values[j], values[j - coins[i - 1]] + 1);
		    }
	    }
	    return values[amount]==INT_MAX-1?-1:values[amount];
    }
```

##硬币问题2
leetcode518(硬币组成种类数)
```
int change(int amount, vector<int>& coins) 
{
        vector<vector<int>> dp(coins.size()+1,vector<int>(amount+1));
        dp[0][0]=1;
        for(int i =1;i<=coins.size();i++)
        {
            dp[i][0]=1;
            for(int j=1;j<=amount;j++)
            {
                if(j<coins[i-1])
                {
                    dp[i][j]=dp[i-1][j];
                }
                else
                {
                    dp[i][j]=dp[i-1][j]+dp[i][j-coins[i-1]];
                }
            }
        }
        return dp[coins.size()][amount];
    }
```

>>dp[i][j]表示使用前i种硬币组成j的组成数量

简化空间:
```
 int change(int amount, vector<int>& coins) 
 {
        vector<int> dp(amount+1);
        dp[0]=1;
        for(int i =1;i<=coins.size();i++)
        {
            for(int j=coins[i-1];j<=amount;j++)
            {
                dp[j]=dp[j]+dp[j-coins[i-1]];
            }
        }
        return dp[amount];
    }
```
