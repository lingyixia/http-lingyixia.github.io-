---
title: 数组常用算法
date: 2019-06-23 12:12:57
category: 算法
tags:
---

#两排序数组最大差值
>>时间复杂度能达到O(n),未排序数组先排序即可

```
int computeMinimumDistanceB(vector<int> a, vector<int> b)
{
    int aindex = 0;
    int bindex = 0;
    int min = abs(a[0] - b[0]);
    while (true)
    {
        if (a[aindex] > b[bindex])
        {
            bindex++;
        } else
        {
            aindex++;
        }
        if (aindex >= a.size() || bindex >= b.size())
        {
            break;
        }
        if (abs(a[aindex] - b[bindex]) < min)
        {
            min = abs(a[aindex] - b[bindex]);
        }
    }
    return min;
}
```

#[找到所有数组中消失的数字](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)
>>要我说这就是个'伪'O(n)

```
vector<int> findDisappearedNumbers(vector<int> &nums)
{
    vector<int> result;
    for (int i = 0; i < nums.size(); ++i)
    {
        if (nums[nums[i] - 1] != nums[i])
        {
            swap(nums[nums[i] - 1], nums[i]);
            i--;
        }
    }
    for (int i = 0; i < nums.size(); ++i)
    {
        if (nums[i] != i + 1)
        {
            result.push_back(i + 1);
        }
    }
    return result;
}
```