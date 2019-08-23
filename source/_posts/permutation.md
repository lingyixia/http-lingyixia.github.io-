---
title: 两种排列对比
date: 2019-03-15 20:54:52
category: 算法
tags: [递归]
---
两种排列对比
<!--more-->

#可重复
```
#include<iostream>
#include<vector>
#include<string>

using namespace std;
vector<char> record = {'a', 'b', 'c'};
vector<vector<char>> result;

void Permutation(vector<char> &current)
{
    if (current.size() == record.size())
    {
        result.push_back(current);
        return;
    }
    for (unsigned int i = 0; i < record.size(); i++)
    {
        current.push_back(record[i]);
        Permutation(current);
        current.pop_back();
    }
}

int main()
{
    vector<char> current;
    Permutation(current);
    return 0;
}
```
>>这里的index表示的是当前current中数据的个数

#不可重复
```
#include<iostream>
#include<vector>
#include<string>

using namespace std;
vector<char> record = {'a', 'b', 'c'};
vector<vector<char>> result;

void Permutation2(vector<char> &current, int index)
{
    if (index == record.size())
    {
        result.push_back(current);
        return;
    }
    for (unsigned int i = index; i < record.size(); i++)
    {
        swap(record[i], record[index]);
        Permutation2(record, index + 1);
        swap(record[i], record[index]);
    }
}

int main()
{
    Permutation2(record, 0);
    return 0;
}
```
>>这里的index表示index之前的保持不变,交换index和他之后的字符