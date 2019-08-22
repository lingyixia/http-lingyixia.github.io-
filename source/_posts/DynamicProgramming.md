---
title: 动态规划系列
date: 2019-04-11 15:46:55
category: 算法
tags: [字符串,背包,递归,硬币]
---

动态规划常用体型,其实最大的难度是遇到一个问题如何将其归类,当归类正确的时候往往是写出的递推公式"似曾相识"的时候。
<!--more-->

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
				dp[i][j] = max(dp[i - 1][j],dp[i][j - 1]);
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

##[最长回文子串](https://leetcode.com/problems/longest-palindromic-substring/)
>>dp[i][j]是记录从i到j是不是回文串,如果是则dp[i][j]=true,反之为false,因为每一次计算dp[i][j]都需要用到dp[i+1][j-1],而j-1>=i+1,即j-i>=2,即i到j的长度至少为3,也就是说间隔长度为1和2的更新不到,因此要先预处理dp,先把间隔为1和2的提前处理
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
str[i+1][j-1] & str[i]=str[j]\\
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

##[最长回文子序列](https://leetcode.com/problems/longest-palindromic-subsequence/)
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
	vector<vector<int>> dp(str.size(), vector<int>(str.size(), 1));
	for (int i = 0; i < str.size()-1; i++)
	{
		if (str[i + 1] == str[i]) dp[i][i + 1]++;
	}
	for (int l = 3; l <= str.size(); l++)
	{
		for (int i = 0; i + l - 1 < str.size(); i++)
		{
			int j = i + l - 1;
			if (str[i] == str[j])
			{
				dp[i][j] = dp[i + 1][j - 1]+2;
			}
			else dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
		}
	}
	return dp[0][str.size()-1];
}
```
##[最长递增子序列](https://leetcode.com/problems/longest-increasing-subsequence/)
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
>>背包问题变种


##[硬币问题1](https://leetcode.com/problems/coin-change/)
最少硬币数量

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

##[硬币问题2](https://leetcode.com/problems/coin-change-2/)
能够组成该数量钱的情况数

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

>>dp[i][j]表示使用前i种硬币组成j的组成数量,dp[i][j]=dp[i-1][j]+dp[i][j-coins[i-1]]的意思，第一项表示不使用币种i组成j的种类数量，第二项表示至少使用一个币种i组成j的种类数，j-coins[i-1]就确保了至少使用一个币种i

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
#数组分裂
##[数组分裂一](https://leetcode.com/problems/partition-equal-subset-sum/)
>>背包问题变种。其实就是数组中每个数字是一个物品,value=weight=数值,看能不能**恰好**装满背包的一半

```
 bool canPartition(vector<int>& nums) 
 {
        int sum = 0;
        for(int i = 0;i<nums.size();i++)
        {
            sum+=nums[i];
        }
        if((sum&1)==1) return false;
        int bag = sum/2;
        vector<int> values(bag+1,INT_MIN);
        values[0]=0;
        for(int i =1;i<=nums.size();i++)
        {
            for(int j = bag;j>=nums[i-1];j--)
            {
                values[j]=max(values[j],values[j-nums[i-1]]+nums[i-1]); 
                if(values[j]==bag) return true;
            }
        }
        return false;
    }
```

##数组分裂二
>>背包问题变种。将数组分为两部分，求各自和的差的绝对值(两部分的和最接近)其实就是数组中每个数字是一个物品,value=weight=数值,target(bag)=sum/2,看这个target(bag)下最大的和是多少

```
int splitArray(vector<int> nums)
{
	int sum = 0;
	for (int i = 0; i < nums.size(); i++)
	{
		sum += nums[i];
	}
	int target = sum/2;
	vector<vector<int>> dp(nums.size() + 1, vector<int>(target + 1));
	for (int i = 1; i <=nums.size(); i++)
	{
		for (int j = 1; j <= target; j++)
		{
			if (nums[i-1]>j)
			{
				dp[i][j] = dp[i-1][j];
			}
			else
			{
				dp[i][j] = max(dp[i-1][j],dp[i-1][j-nums[i-1]]+nums[i-1]);
			}
		}
	}
	return ( sum- target) - dp[nums.size()][target];
}
```

空间优化:

```
int splitArray(vector<int> nums)
{
	int sum = 0;
	for (int i = 0; i < nums.size(); i++)
	{
		sum += nums[i];
	}
	int target = sum/2;
	vector<int> dp(target + 1);
	for (int i = 1; i <=nums.size(); i++)
	{
		for (int j = target; j >= nums[i - 1]; j--)
		{
			dp[j] = max(dp[j],dp[j-nums[i-1]]+nums[i-1]);
		}
	}
	return ( sum- target) - dp[target];
}
```

#[零和问题](https://leetcode.com/problems/ones-and-zeroes/)
>>背包问题变种，难点在于分清**资源**和**目标**,此题有两个**目标**

```
int findMaxForm(vector<string> strs, int m, int n) 
{
	vector<vector<vector<int>>> dp(strs.size() + 1, vector<vector<int>>(m + 1, vector<int>(n + 1)));
	for(int k = 1;k<=strs.size();k++)
	{
		int zeros = count(strs[k-1].begin(), strs[k-1].end(), '0');
		int ones = count(strs[k-1].begin(), strs[k-1].end(), '1');
		for (int i = 0; i <= m; i++)//注意从零开始
		{
			for (int j = 0; j <= n; j++)//注意从零开始
			{
				if (i< zeros || j<ones)
				{
					dp[k][i][j] = dp[k-1][i][j];
				}
				else
				{
					dp[k][i][j] = max(dp[k-1][i][j], dp[k-1][i - zeros][j - ones] + 1);
				}
			}
		}
	}
	return dp[strs.size()][m][n];
}
```

空间优化:

```
int findMaxForm(vector<string>& strs, int m, int n) 
{
    vector<vector<int>> dp(m+1,vector<int>(n+1));
    for(string str:strs)
    {
        int ones = count(str.begin(),str.end(),'1');
        int zeros = count(str.begin(),str.end(),'0');
        for(int i = m;i>=zeros;i--)
        {
            for(int j = n;j>=ones;j--)
            {
                dp[i][j]=max(dp[i][j],dp[i-zeros][j-ones]+1);
            }
        }
    }
    return dp[m][n];
}
```

#[编辑距离](https://leetcode.com/problems/edit-distance/description/)
>>dp[i][j]表示将word1[1:i]转为word2[1:j]所需要的最少次数
$$
dp[i][j] = \begin{cases}
dp[i-1][j-1] & word1[i]==word2[j] \\
min(dp[i-1][j],dp[i][j],dp[i][j-1]) & word1[i]!=word2[j]
\end{cases}
$$
dp[i-1][j]、dp[i][j]、dp[i][j-1]表示三种转换方式，取其最小.
dp[i-1][j]+1表示把word1[1:i-1]转为word2[1:j],然后删除words[i]
dp[i-1][j-1]+1表示把word1[1:i-1]转为word2[1:j-1]然后把word1[i]替换为word2[j]
dp[i][j-1]+1表示把word1[1:i]转为dp[1:j-1]然后把word2[j]插入到末尾

```
template <typename T>
T min(T a,T b,T c)
{
    if(c>b) c=b;
    if(c>a) c=a;
    return c;
}
int minDistance(string word1, string word2) 
{
    vector<vector<int>> dp(word1.size()+1,vector<int>(word2.size()+1));
    for(int i=1;i<=word1.size();i++)
    {
        dp[i][0]=i;
    }
    for(int i=1;i<=word2.size();i++)
    {
        dp[0][i]=i;
    }
    for(int i =1;i<=word1.size();i++)
    {
        for(int j =1;j<=word2.size();j++)
        {
            if(word1[i-1]==word2[j-1])
            {
                dp[i][j]=dp[i-1][j-1];
            }
            else
            {
                dp[i][j]=min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1;
            }
        }
    }
    return dp[word1.size()][word2.size()];
}
```

#[矩阵中最大正方形](https://leetcode.com/problems/maximal-square/description/)
>>同编辑距离.dp[i][j]记录以matrix[i][j]结尾的边长(要包含matrix[i][j])

$$
dp[i][j] = \begin{cases}
dp[i+1][j-1]+2 & str[i]=str[j] \\
max(dp[i+1][j],dp[i][j-1]) & others
\end{cases}
$$
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
				maxResult= max(maxResult, dp[i][j]* dp[i][j]);
			}
		}
	}
}
```
#[鹰蛋问题](https://leetcode.com/problems/super-egg-drop/)
>>问题大概就是一栋楼N层,你有K个一模一样的鸡蛋,问如何找到一个最高层,这个层扔下去刚好鸡蛋不碎,在上一层就碎了。
此题变种较多,在编程角度是一个动态规划，在数学角度是一个趣味问答题，下面具体分析.
数学角度:
比如我们只有两枚蛋,让你给出一个方案，这个方案测试的次数最少，而且一定能找到这个楼层.
难度在于思维转换,比如我们现在已经找到了这个次数X，不可能有某个方案的测试次数比X更小，那么X是多少呢?
首先我们要转换的思维是:某个楼层,X是它最少的测试次数,意味着这个楼层是X能测试得到的最高楼层.所以我们把题转为这个最高楼层是多少。
现在我们想一下,两个鸡蛋最少得有两次测试机会吧，那么如果X是2，这个最高楼层是多少呢?
想想方案,如果第一个蛋测试的是5楼，咔，破了,这时候你只剩下一个蛋,只有一次机会了,但是这次机会你只能测试第一层啊,如果没破，机会没了,没测出来.也就是说,只有两次机会,你第一次不能测的太高,最高只能测试第2层.那最低呢?不需要最低，要得到最高楼层只需要考虑每个蛋放的最高楼层是多少才能得到最大效益，也就是说，如果最少次数是X，那第一次放置的最高楼层只能是X,这样才能保证在第一次破的时候剩下的蛋在X-1次内完成测试。那第二次呢?还剩X-1次机会那就在向上走X-1层呗，，，以此类推直到最顶层。
也就是说，如果楼层是$F$，那最少次数只需要解方程:
$$
X+(X-1)+(X-2)+...+1=\frac{X(X-1)}{2}≤F
$$
也就是说两个鸡蛋,X次机会最多可测试的楼层数是$\frac{X(X-1)}{2}$
那么三个鸡蛋X次的最大楼层是多少呢?那么同样，第一次首先要确定一个最高楼层，这个楼层下如果破了，剩下的2个蛋和X-1次机会能恰好测完.那么就可以这么想，第一次放在第f层，那剩下的2个蛋和X-1次机会最高到$\frac{X(X-1)}{2}$,所以f就是$\frac{(X-1)(X-2)}{2}$+1，至于为什么加1稍微想想就知道了，剩下的就一样了.(看到没，这题还可以改成如果n个蛋，X次机会，能测试的最高楼层是多少.)
动态规划角度:同样利用上诉思维转换dp[i][j]表示j个鸡蛋i次机会能确定的最高楼层
dp[l][i] = dp[l - 1][i - 1] + dp[l - 1][i] + 1这个地推公式的含义
dp[l][i]:i个鸡蛋l次机会所能确定的最高楼层
dp[l-1][i-1]:第一枚鸡蛋的最高位置
dp[l - 1][i]:剩下鸡蛋能在此基础上累加的高度。
其实上面这个公式是这个:
dp[l][i] = max(dp[l - 1][i - 1], dp[l - 1][i - 1]+ dp[l - 1][i] + 1)
前面代表第一个碎了，后面代表第一个没碎，因为前面肯定小于等于后面，所以就不写了.

```
 int superEggDrop(int K, int N)
 {
	vector<vector<int>> dp(N+1,vector<int>(K+1));//dp[i][j]表示j个鸡蛋i次机会能够确定的最高楼层
	int l = 0;
	while (dp[l][K] < N)
	{
		l++;
		for (int i = 1; i <= K; i++)
		{
			dp[l][i] = dp[l - 1][i - 1] + dp[l - 1][i] + 1;
		}
	}
	return l;
}
```

#其他DP
##连续数组最大和
>>给出一个数组，求连续数组最大和.
//递推公式为DP[i] = max{DP[i-1] + A[i],DP[i]},对于当前点i,要么与前面连起来组成和,要么自己组成和

```
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
##连续数组最大积
>>与最大和不同点在于数组中的负数会对积有影响，最大和只需要记录当前最大和即可，但是最大积需要记录最大积和最小积,举例说明:当前数字是-1，之前最大积是5,最小积是-6,那么到此时最大积应当是-1 * -6=6而不是-1.

```
int maxProduct(vector<int> &nums)
{
    int result = INT_MIN;
    int beforeMaxProduct = 1;
    int beforeMinProduct = 1;
    for (int i = 0; i < nums.size(); ++i)
    {
        int tempMax = beforeMaxProduct * nums[i];
        int tempMin = beforeMinProduct * nums[i];
        beforeMaxProduct = max(max(tempMax, tempMin), nums[i]);
        beforeMinProduct = min(min(tempMax, tempMin), nums[i]);
        result = max(beforeMaxProduct, result);
    }
    return result;
}
```
##数组最大积(不一定连续)

```
int maxProduct(vector<int> &nums)
{
    int result = INT_MIN;
    int beforeMaxProduct = 1;
    int beforeMinProduct = 1;
    for (int i = 0; i < nums.size(); ++i)
    {
        int tempMax = beforeMaxProduct * nums[i];
        int tempMin = beforeMinProduct * nums[i];
        beforeMaxProduct = max(max(tempMax, tempMin), beforeMaxProduct);
        beforeMinProduct = min(min(tempMax, tempMin), beforeMinProduct);
        result = max(beforeMaxProduct, result);
    }
    return result;
}
```

##最大子矩阵和
>>和上面基本一样

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

##[小偷](https://leetcode.com/problems/house-robber)

```
int rob(vector<int> &nums)
{
    if (nums.size() == 0)
    {
        return 0;
    } else if (nums.size() == 1)
    {
        return nums.back();
    } else if (nums.size() == 2)
    {
        return max(nums.front(), nums.back());
    } else
    {
        vector<int> dp(nums.size());
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for (int i = 2; i < nums.size(); ++i)
        {
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp.back();
    }
}
```

##[Jump Game](https://leetcode.com/problems/jump-game/)

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