---
title: 两种排列对比
date: 2019-03-15 20:54:52
category: 算法
tags: [递归]
---
#可重复
```
vector<char> record = { 'a','b','c' };
vector<vector<char>> result;
void Permutation1(vector<char>& current,int index)
{
	if (index == record.size())
	{
		result.push_back(current);
		return;
	}
	for (int i = 0; i < record.size(); i++)
	{
		current.push_back(record[i]);
		Permutation1(current,index+1);
		current.pop_back();
	}
}
int main()
{
	vector<char> current;
	Permutation1(current,0);
	return 0;
}
```

#不可重复
```
#include<iostream>
#include<vector>
#include<string>
using namespace std;
vector<char> record = { 'a','b','c' };
vector<vector<char>> result;

void swap(vector<char>& record,int i ,int j)
{
	int temp = record[i];
	record[i] = record[j];
	record[j] = record[i];
}
void Permutation2(vector<char>& current, int index)
{
	if (index == record.size())
	{
		result.push_back(current);
		return;
	}
	for (unsigned int i = index; i < record.size(); i++)
	{
		swap(record, i, index);
		Permutation2(record, index + 1);
		swap(record, i, index);
	}
}
int main()
{
	vector<char> current;
	Permutation2(current, 0);
	return 0;
}
```

