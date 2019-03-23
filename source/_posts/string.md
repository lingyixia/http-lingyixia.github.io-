---
title: 字符串问题
date: 2019-02-22 21:35:47
category: 算法
tags: [string]
---

#最长公共字串
>>[其实很简单](https://www.cnblogs.com/guolipa/p/10053551.html),动态规划,记录矩阵record[i][j]表示以s[i]和t[j]结尾的最长公共字串,可以看出如果s[i]!=t[j]则record[i][j]一定是0,因为两者不相等,以他们为结尾不可能有公共字串,只有s[i]==t[j]的时候才不为0

```
string commonString(string s1, string s2)
{
	int maxLength=0;
	int index;
	vector<vector<int>> record(s1.length(), vector<int>(s2.length()));
	for (int i = 0; i < s1.size(); i++)
	{
		for (int j = 0; j < s2.size(); j++)
		{
			if (s1[i] == s2[j])
			{
				if (i == 0 || j == 0) record[i][j] == 1;
				else record[i][j] = record[i - 1][j - 1]+1;
			}
			if (record[i][j] > maxLength)
			{
				maxLength = record[i][j];
				index = i;
			}
		}
	}
	return s1.substr(index - maxLength+1,maxLength);
}
```