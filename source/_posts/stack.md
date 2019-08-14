---
title: 栈
date: 2019-02-08 16:22:29
catagory: 算法
tags: [栈]
---
# 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
<!--more-->
算法描述: 只需要按照顺序走一遍即可
```
 bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        stack<int> s;
        int index=0;
        for(int i =0;i<pushV.size();i++)
        {
            s.push(pushV[i]);
            while(!s.empty() && s.top()==popV[index])
            {
                s.pop();
                index++;
            }
        }
        return s.empty();
    }
```

Tip: 对于一个入栈顺序的弹栈序列必然有这么一个特征:
出栈序列中每个数后面的比它小的数必然按照降序排列
比如入栈顺序是:1,2,3,4
1. 4,1,2,3不可能是出栈顺序,因为4后面比4小的数1,2,3不是降序排列
2. 3,1,4,2也不合法,3后面比3小的数1,2不是降序排列
3. 1,2,3,4合法,当前每个数后面没有比它小的

#[删除相邻重复字符串](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/)

```
string removeDuplicates(string S)
{
    stack<char> s;
    for (auto ch:S)
    {
        if (s.empty() || ch != s.top())
        {
            s.push(ch);
        } else
        {
            s.pop();
        }
    }
    string result = "";
    while (!s.empty())
    {
        result = s.top()+result;
        s.pop();
    }
    return result;
}
```