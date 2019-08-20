---
title: 二分法扩展
dat : 2019-02-24 16:13:40
category: 算法
tags: [二分法]
---

思路很简单，细节最致命，不要看不起看似很简单的二分法，不同的细节决定了它不同的用途!!!
<!--more-->

#普通二分查找
>>找到则返回下标，找不到返回-1

```
int search(vector<int>& nums,int target)
{
	int low = 0;
	int high = nums.size()-1;
	int mid = 0;
	while(low<=high)
	{
		mid = (low+high)/2;
		if(target<nums[mid])
		{
			high=mid-1;
		}
		else if (target>nums[mid])
		{
			low=mid+1;
		}
		else
		{
			return mid;
		}
	}
	return -1;
}
```

#查找插入位置
>>给出一个升序数组,一个数字，从数组中找出该数字的插入位置,可以这样想:循环结束的条件(假设插入的数字数组中没有)是low=high+1,此时target必然array[high] < target < array[low],因此插入位置一定是low处

```
 int searchInsert(vector<int>& nums, int target) {
	int low = 0;
	int high = nums.size()-1;
	int middle = 0;
	while (low <= high)
	{
		middle = (low + high) / 2;
		if (nums[middle] == target)
		{
			return middle;
		}
		else if (target > nums[middle])
		{
			low = middle + 1;
		}
		else
		{
			high = middle - 1;
		}
	}
	return low;
}
```

#查找上下界
>>给出一个升序数组(可能有重复),一个数字,从数组中找出该数字的下界

```
int searchRange(vector<int>& nums, int target)
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
	if(low>=nums.size() || nums[low]!=target)
	    return -1;
	return low;
}
```

>>给出一个升序数组(可能有重复),一个数字,从数组中找出该数字的上界

```
int searchRange(vector<int>& nums, int target)
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
	if(high<0 || nums[high]!=target)
	    return -1;
	return high;
}
```