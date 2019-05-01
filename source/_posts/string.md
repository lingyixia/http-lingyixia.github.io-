---
title: 字符串问题
date: 2019-02-22 21:35:47
category: 算法
tags: [string]
---

#最长公共子串
>>[其实很简单](https://www.cnblogs.com/guolipa/p/10053551.html),动态规划,记录矩阵record[i][j]表示以s[i]和t[j]结尾的最长公共字串长度,可以看出如果s[i]!=t[j]则record[i][j]一定是0,因为两者不相等,以他们为结尾不可能有公共字串,只有s[i]==t[j]的时候才不为0

```
string maxLengthStr(string s1, string s2)
{
	int maxLength = 0;
	int index;
	vector<vector<int>> record(s1.size()+1, vector<int>(s2.size()+1));
	for (int i = 1; i <= s1.size(); i++)
	{
		for (int j = 1; j <= s2.size(); j++)
		{
			if (s1[i-1] == s2[j-1])
			{
				record[i][j] = record[i - 1][j - 1] + 1;
			}
			if (record[i][j] > maxLength)
			{
				maxLength = record[i][j];
				index = i;
			}
		}
	}
	return s1.substr(index - maxLength, maxLength);
}
```

#最长公共子序列
>>跟上面没有什么不同,动态规划,记录矩阵record[i][j]表示以s[i]和t[j]结尾的最长公共子序列长度,可以看出如果s[i]!=t[j]则record[i][j]= max(record[i-1][j],record[i][j-1]),这里是与上面dp公式唯一不同点,而当s[i]=t[j]的时候record[i][j]=record[i-1][j-1]+1

```
int maxLengthSequence(string s1, string s2)
{
	vector<vector<int>> records(s1.size()+1, vector<int>(s2.size()+1));
	for (int i = 1; i <= s1.size(); i++)
	{
		for (int j = 1; j <= s2.size(); j++)
		{
			if (s1[i-1] == s2[j-1])
			{
				records[i][j] = records[i - 1][j - 1]+1;
			}
			else
			{
				records[i][j] = max(records[i - 1][j],records[i][j - 1]);
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
		else if (records[i][j-1] >= records[i-1][j]) j--;
		else i--;
	}
	return records[s1.size()][s2.size()];
}
```

#最长回文子串
>>递推公式为:records[i][j]=records[i+1][j-1] if str[i][j],初始状态为records[i][i]=1,records[i][i+1]=2 if str[i]=str[i+1].
records[i][j]是记录从i到j是不是回文串,如果是则records[i][j]=true,反之为false,因为每一次计算records[i][j]都需要用到records[i+1][j-1],而j-1>=i+1,即j-i>=2,即i到j的长度至少为3,也就是说间隔长度为1和2的更新不到,因此要先预处理records,先把间隔为1和2的提前处理

```
string longestPalindrome(string str)
{
	int index = 0;
	int maxLength = 1;
	vector<vector<bool>> records(str.size(), vector<bool>(str.size(), false));
	for (int i = 0; i <= str.size(); i++)
	{
		records[i][i] = true;
		if (i + 1 < str.size())
		{
			if (str[i] == str[i + 1])
			{
				records[i][i + 1] = true;
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
			if (str[i] == str[j] && records[i + 1][j - 1])
			{
				records[i][j] = true;
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

#最长回文子序列
>>跟上诉差不多,公式为records[i][j] = records[i + 1][j - 1]+2 if str[i] = str[j],records[i][j] = max(records[i + 1][j], records[i][j - 1]) if str[i] != str[j]

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
#最长递增子序列
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
