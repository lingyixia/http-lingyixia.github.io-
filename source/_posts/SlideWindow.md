---
title: 滑动窗口系列
date: 2019-05-11 17:04:10
category: 算法
tags:
---
滑动窗口
<!--more-->

#最长连续不重复字串
>>[leetcode3](https://leetcode.com/problems/longest-substring-without-repeating-characters/)其实思想就是用一个窗口[left,right]表示该窗口中没有重复数字,那么下一步要窗口右移,即第right+1个,如果第right+1个在[left,right]中,那么就找出其坐标,使left更新为该坐标,否则加入窗口即可,这是法二的思维,法一的思维是如果在窗口中就不断右移left,直到该坐标位置,其实思想是一样的.

##法一
```
int lengthOfLongestSubstring(string s)
{
    if(s.empty()) return 0;
	unordered_set<char> records;
	int left = 0;
	int right = -1;
	int maxLength = INT_MIN;
	while (left<s.size())
	{
		if (right+1<s.size() && records.count(s[right+1])==0)
		{
			records.insert(s[right+1]);
			right++;
		}
		else
		{
			records.erase(s[left]);
			left++;
		}
		maxLength = max(maxLength,right-left+1);
	}
	return maxLength;
}
```

##法二
```
int lengthOfLongestSubstring(string s)
{
	unordered_map<char,int> records;
	int maxLength = INT_MIN;
	for (int right = 0,left=0; right < s.size(); right++)
	{
		if (records.count(s[right])>0)
		{
			left = max(records[s[right]], left);
		}
		maxLength = max(maxLength,right-left+1);
		records[s[right]] = right+1;
	}
	return maxLength;
}
```