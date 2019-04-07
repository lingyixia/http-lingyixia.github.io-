---
title: 字符串问题
date: 2019-02-22 21:35:47
category: 算法
tags: [string]
---

#最长公共字串
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