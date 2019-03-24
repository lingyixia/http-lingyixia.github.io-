---
title: 二分法扩展
dat : 2019-02-24 16:13:40
category: 算法
tags: [二分法]
---
>>给出一个升序数组(有重复),一个数字,从数组中找出该数字的上下界

```
void searchRange(vector<int>& nums, int target)
{
	int low = 0;
	int high = nums.size() - 1;
	int mid = 0;
	while (low <= high)
	{
		mid = (low + high) / 2;
		if (target <= nums[mid])
		{
			high = mid - 1;
		}
		else
		{
			low = mid + 1;
		}
	}
}
```
>>最终得到的结果low是下界,high是low-1

```
void searchRange(vector<int>& nums, int target)
{
	int low = 0;
	int high = nums.size() - 1;
	int mid = 0;
	while (low <= high)
	{
		mid = (low + high) / 2;
		if (target >= nums[mid])
		{
			low = mid + 1;
		}
		else
		{
			high = mid - 1;
		}
	}
}
```
>>最终得到的结果high是上界,low是high+1