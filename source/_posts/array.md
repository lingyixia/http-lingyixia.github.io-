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
#有序数组并集交集

```
#include <iostream>
#include <queue>

using namespace std;

vector<int> aggregate(vector<int> A, vector<int> B)
{
    vector<int> result;
    int pointA = 0;
    int pointB = 0;
    while (pointA < A.size() && pointB < B.size())
    {
        if (A[pointA] < B[pointB])
        {
            result.push_back(A[pointA++]);
        } else if (A[pointA] > B[pointB])
        {
            result.push_back(B[pointB++]);
        } else
        {
            result.push_back(A[pointA++]);
            pointB++;
        }
    }
    return result;
}

vector<int> intersection(vector<int> A, vector<int> B)
{
    vector<int> result;
    int pointA = 0;
    int pointB = 0;
    while (pointA < A.size() && pointB < B.size())
    {
        if (A[pointA] < B[pointB])
        {
            pointA++;
        } else if (A[pointA] > B[pointB])
        {
            pointB++;
        } else
        {
            result.push_back(A[pointA]);
            pointA++;
            pointB++;
        }
    }
    return result;
}

int main()
{
    vector<int> A = {1, 3, 4, 5, 7};
    vector<int> B = {2, 3, 5, 8, 9};
    vector<int> result = aggregate(A, B);
    for (auto r:result)
    {
        cout << r << endl;
    }
    return 0;
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

#[Ksum](https://blog.csdn.net/Gease_Gg/article/details/82055295)
>>数组中K个数相加为target(不定几个用背包)

```
class Solution {
public:
    vector<vector<int>> result;

void kSum(vector<int> &nums, int k, int start, int end, int target, vector<int> current)
{
    if (k == 2)
    {
        int low = start;
        int high = end;
        while (low < high)
        {
            if (nums[low] + nums[high] == target)
            {
                current.push_back(nums[low]);
                current.push_back(nums[high]);
                result.push_back(current);
                while (low < end && nums[low + 1] == nums[low]) low++;
                while (low < end && nums[high - 1] == nums[high]) high--;
                low++;
                high--;
                current.pop_back();
                current.pop_back();
            } else if (nums[low] + nums[high] < target)
            {
                low++;
            } else
            {
                high--;
            }
        }
    } else
    {
        for (int i = start; i <= end - (k - 1); ++i)
        {
            if (i > start && nums[i] == nums[i - 1])continue;
            current.push_back(nums[i]);
            kSum(nums, k - 1, i + 1, end, target - nums[i], current);
            current.pop_back();
        }
    }
}

vector<vector<int>> fourSum(vector<int> &nums, int target)
{
    vector<int> current;
    sort(nums.begin(), nums.end());
    kSum(nums, 4, 0, nums.size() - 1, target, current);
    return result;
}
};
```

#[数组中右面第一个比它大的距离](https://leetcode.com/problems/daily-temperatures/submissions/)

```
 vector<int> dailyTemperatures(vector<int> T)
{
    vector<int> result(T.size());
    stack<int> record;
    int pos = 0;
    while (pos < T.size())
    {
        if (record.empty())
        {
            record.push(pos++);
        } else
        {
            if (T[pos] <= T[record.top()])
            {
                record.push(pos++);
            } else
            {
                result[record.top()] = pos-record.top();
                record.pop();
            }
        }
    }
    return result;
}
```
