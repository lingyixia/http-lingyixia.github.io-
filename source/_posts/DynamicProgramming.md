---
title: 动态规划系列
date: 2019-04-11 15:46:55
category: 算法
tags: [字符串,背包,递归,硬币]
---

#字符串问题
##最长公共子串
>>[其实很简单](https://www.cnblogs.com/guolipa/p/10053551.html),动态规划,记录矩阵dp[i][j]表示以s1[i]和s2[j]结尾的最长公共字串长度,可以看出如果s1[i]!=s2[j]则dp[i][j]一定是0,因为两者不相等,以他们为结尾不可能有公共字串,只有s1[i]==s2[j]的时候才不为0,即:
$$
dp[i][j]=\begin{cases}
0 & s1[i]!=s2[j] \\
dp[i-1][j-1]+1 & s1[i]=s2[j] \\
\end{cases}
$$

```
string maxLengthSubstr(string s1, string s2)
{
	int maxLength = 0;
	int index;
	vector<vector<int>> dp(s1.size()+1, vector<int>(s2.size()+1));
	for (int i = 1; i <= s1.size(); i++)
	{
		for (int j = 1; j <= s2.size(); j++)
		{
			if (s1[i-1] == s2[j-1])
			{
				dp[i][j] = dp[i - 1][j - 1] + 1;
			}
			if (dp[i][j] > maxLength)
			{
				maxLength = dp[i][j];
				index = i;
			}
		}
	}
	return s1.substr(index - maxLength, maxLength);
}
```

##最长公共子序列
>>跟上面没有什么不同,动态规划,记录矩阵dp[i][j]表示以s1[i]和s2[j]结尾的最长公共子序列长度,可以看出如果s1[i]!=s2[j]则dp[i][j]= max(dp[i-1][j],dp[i][j-1]),这里是与上面dp公式唯一不同点,而当s1[i]=s2[j]的时候dp[i][j]=dp[i-1][j-1]+1,即:

$$
dp[i][j]=\begin{cases}
max(dp[i-1][j],dp[i][j-1]) & s1[i]!=s2[j] \\
dp[i-1][j-1]+1 & s1[i]=s2[j] \\
\end{cases}
$$

```
int maxLengthSequence(string s1, string s2)
{
	vector<vector<int>> dp(s1.size()+1, vector<int>(s2.size()+1));
	for (int i = 1; i <= s1.size(); i++)
	{
		for (int j = 1; j <= s2.size(); j++)
		{
			if (s1[i-1] == s2[j-1])
			{
				dp[i][j] = dp[i - 1][j - 1]+1;
			}
			else
			{
				dp[i][j] = max(dp[i - 1][j],dp[i][j - 1);
			}
		}
	}
	for (int i =s1.size(),j=s2.size(); i>=1 && j>=1;)
	{
		if (s1[i - 1] == s2[j - 1])
		{
			cout << s1[i - 1]<<" ";
			i--;
			j--;
		}
		else if (dp[i][j-1] >= dp[i-1][j]) j--;
		else i--;
	}
	return dp[s1.size()][s2.size()];
}
```

##最长回文子串
>>dp[i][j]是记录从i到j是不是回文串,如果是则dp[i][j]=true,反之为false,因为每一次计算dp[i][j]都需要用到dp[i+1][j-1],而j-1>=i+1,即j-i>=2,即i到j的长度至少为3,也就是说间隔长度为1和2的更新不到,因此要先预处理records,先把间隔为1和2的提前处理
递推公式为:
$$
dp[i][j] = \begin{cases}
true & str[i]=str[j] and dp[i+1][j-1]=true\\
false & others
\end{cases}
$$
或
$$
dp[i][j] = \begin{cases}
str[i-1][j-1] & str[i]=str[j]\\
false & others
\end{cases}
$$

```

string longestPalindrome(string str)
{
	int index = 0;
	int maxLength = 1;
	vector<vector<bool>> dp(str.size(), vector<bool>(str.size(), false));
	for (int i = 0; i <= str.size(); i++)
	{
		dp[i][i] = true;
		if (i + 1 < str.size())
		{
			if (str[i] == str[i + 1])
			{
				dp[i][i + 1] = true;
				index = i;
				maxLength = 2;
			}
		}
	}
	for (int l = 3; l <= str.size(); l++)
	{
		for (int i = 0; i + l - 1 < str.size(); i++)
		{
			int j = i + l - 1;
			if (str[i] == str[j] && dp[i + 1][j - 1])
			{
				dp[i][j] = true;
				if (l > maxLength)
				{
					index = i;
					maxLength = l;
				}
			}
		}
	}
	return str.substr(index, maxLength);
}
```

##最长回文子序列
>>跟上诉差不多,公式为:
$$
dp[i][j] = \begin{cases}
dp[i+1][j-1]+2 & str[i]=str[j] \\
max(dp[i+1][j],dp[i][j-1]) & others
\end{cases}
$$

```
int longestPalindromeSubseq(string str)
{
	vector<vector<int>> records(str.size(), vector<int>(str.size(), 1));
	for (int i = 0; i < str.size()-1; i++)
	{
		if (str[i + 1] == str[i]) records[i][i + 1]++;
	}
	for (int l = 3; l <= str.size(); l++)
	{
		for (int i = 0; i + l - 1 < str.size(); i++)
		{
			int j = i + l - 1;
			if (str[i] == str[j])
			{
				records[i][j] = records[i + 1][j - 1]+2;
			}
			else records[i][j] = max(records[i + 1][j], records[i][j - 1]);
		}
	}
	return records[0][str.size()-1];
}
```
##最长递增子序列
>>dp解法一,dp[i]为以第i个元素结尾的最长递增子序列长度,$O(n^2)$


```
int lengthOfLIS(vector<int>& nums) 
{
        if(nums.empty()) return 0;
        int result=1;
        vector<int> dp(nums.size(),1);
        for(int i =1;i<nums.size();i++)
        {
            for(int j =0;j<i;j++)//不断更新dp[i]的值
            {
                if(nums[i]>nums[j])
                {
                    dp[i]=max(dp[j]+1,dp[i]);
                }
            }
            result=max(dp[i],result);
        }
        return result;
    }
```

#背包问题
>0-1背包、完全背包、多重背包问题简记.一般而言,背包问题有两种:**要求恰好装满**和**不要求恰好装满**,其实两者**仅仅**初始化数组不同,前者要初始化`values[i][0]=0`,其他全为`-INT_MIN`.后者初始化全为0即可。

##0-1背包问题
###常规写法
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
###空间优化写法
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
###再次改善
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

##完全背包问题
###常规写法
与0-1背包唯一的不同就是递推公式：`values[i][j] = max(values[i - 1][j], values[i][j - woods[i-1].volume] + woods[i-1].value);`不在详细说明

###空间优化写法
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

##多重背包问题（即每种物品的数量有上限）
###常规写法
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
###空间优化写法
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

#硬币问题
##硬币问题1
leetcode322(最少硬币数量)
```
int coinChange(vector<int>& coins, int amount) 
{
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

#矩阵中最大正方形
>>dp[i][j]记录以matrix[i][j]结尾的边长(要包含matrix[i][j])
```
template <typename T>
T min(T a, T b, T c)
{
	if (c>a)
	{
		c = a;
	}
	if (c>b)
	{
		c = b;
	}
	return c;
}
int maximalSquare(vector<vector<char>>& matrix)
{
	int maxResult = 0;
	vector<vector<int>> dp(matrix.size(),vector<int>(matrix[0].size()));
	for (int i = 0; i < matrix.size(); i++)
	{
		if (matrix[i][0]=='1')
		{
			dp[i][0] = 1;
			maxResult = max(maxResult, 1);
		}
	}
	for (int i = 0; i < matrix[0].size(); i++)
	{
		if (matrix[0][i] == '1')
		{
			dp[0][i] = 1;
			maxResult = max(maxResult, 1);
		}
	}
	for (int i = 1; i < dp.size(); i++)
	{
		for (int j = 1; j < dp[0].size(); j++)
		{
			if (matrix[i][j]=='0')
			{
				dp[i][j] = 0;
			}
			else
			{
				dp[i][j] = min(dp[i-1][j-1],dp[i][j-1],dp[i-1][j])+1;
				maxResult= maxResult = max(maxResult, dp[i][j]* dp[i][j]);
			}
		}
	}
}
```