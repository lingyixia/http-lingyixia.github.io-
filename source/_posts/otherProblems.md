---
title: otherProblems
date: 2019-05-29 16:21:08
catrgory: 算法
tags:
---

#[大数相乘](https://blog.csdn.net/kangkanglou/article/details/79894208)

```
string multiply(string num1, string num2)
{
    vector<int> record(num1.size() + num2.size());
    for (int i = num1.size() - 1; i >= 0; --i)
    {
        for (int j = num2.size() - 1; j >= 0; --j)
        {
            int mul = (num1[i] - '0') * (num2[j] - '0');
            int pos1 = i + j;
            int pos2 = i + j + 1;
            int temp = mul + record[pos2];
            record[pos1] += (temp / 10);
            record[pos2] = temp % 10;
        }
    }
    string result = "";
    bool flag = false;
    for (auto num:record)
    {
        if (num != 0) flag = true;
        if (flag)
            result += to_string(num);
    }
    return result.empty() ? "0" : result;
}
```