---
layout: post
title: 排序算法
category: 算法
tags: [排序]
date: 2018-11-20 10:11:08
description: 各种排序算法总结
---
>排序算法大概分为:交换类(快速排序和冒泡排序),插入类(简单插入排序和希尔排序),选择类(简单选择排序和堆排序),归并类(二路归并排序和多路归并排序),以下算法全部以增序为例

#交换类排序:
##冒泡排序
>>一共进行$n-1$轮次(最后一轮没必要,因为就剩一个了顺序已经排好),每一轮把第$[0,1,2,3...,i]$小的上浮。

实例代码:
```
#include<iostream>
#include<vector>
#include<string>
#include<ctime>
using namespace std;
void bubbleSort(vector<int>& array);
int main()
{
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	vector<int> array = {5, 1, 9, 3, 7, 4, 8, 6, 2 };
	bubbleSort(arrray);
	endTime = clock();//计时结束
	cout << "总计时长" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	return 0;
}
void bubbleSort(vector<int>& array)
{
	bool flag = true;//flag为false的意义为i之后的已经不需要排序了
	for (int i = 0; i < array.size() && flag; i++)
	{
		flag = false;
		for (int j = array.size() - 2; j >= i; j--)
		{
			if (array[j] > array[j + 1])
			{
				swap(array[j], array[j + 1]);
				flag = true;
			}
		}
	}
}
```
总结分析:

|名称|时间|最好|最坏|空间|稳定|备注|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|冒泡排序|$O(n^2)$|$O(n)$|$O(n^2)$|$O(1)$|是|每一轮都有一个元素到最终位置上|

##快速排序

>>把一个序列,选一个数(第一个数)进行划分。左边小于x,中间x,右边大于x。再依次递归划分左右两边。

实例代码:
```
#include<iostream>
#include<vector>
#include<string>
#include<ctime>
using namespace std;

void Qsort(vector<int>& array,int low,int high);//快速排序
int Partition(vector<int>& array, int low, int high);
int main()
{
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	vector<int> array = {5, 1, 9, 3, 7, 4, 8, 6, 2 };
	Qsort(array,0,array.size()-1);
	endTime = clock();//计时结束
	cout << "总计时长" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	return 0;
}
void Qsort(vector<int>& array, int low, int high)
{
	int privot;
	if (low<high)
	{
		privot = Partition(array, low, high);
		Qsort(array,low,privot-1);
		Qsort(array, privot + 1, high);
	}
}
int Partition(vector<int>& array, int low, int high)
{
	int privotKey = array[low];
	int finalPos = low;
	while (low<high)
	{
		while (low<high&&array[high]>=privotKey)
		{
			high--;
		}
		array[low] = array[high];
		while (low < high&&array[low] <= privotKey)
		{
			low++;
		}
		array[high] = array[low];
	}
	array[low] = privotKey;
	return low;
}
```
总结分析:

|名称|时间|最好|最坏|空间|稳定|备注|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|快速排序|$O(nlogn)$|$O(nlogn)$|$O(n^2)$|栈的深度$O(log_2n)$|否|基本有序或者基本逆序，效果最差

#插入排序
##直接插入排序
>>前面的已经有序,把后面的插入到前面有序的元素中,不断增长有序序列。
1.找到待插入位置
2.给插入元素腾出空间,边比较边移动

```
#include<iostream>
#include<vector>
#include<string>
#include<ctime>
using namespace std;

void directInsertSort(vector<int>& array);
int main()
{
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	vector<int> array = {5, 1, 9, 3, 7, 4, 8, 6, 2 };
	directInsertSort(array);
	endTime = clock();//计时结束
	cout << "总计时长" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	return 0;
}
void directInsertSort(vector<int>& array)
{
	for (int i = 1; i < array.size(); i++)
	{
		int temp = array[i];
		int j = i - 1;
		while (j >= 0 && temp < array[j])
		{
			array[j + 1] = array[j];
			j--;
		}
		array[j + 1] = temp;
	}
}
```
总结分析:

|名称|时间|最好|最坏|空间|稳定|备注|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|直接插入排序|$O(n^2)$|$O(n)$|$O(n^2)$|$O(1)$|是|适用于顺序存储|

##希尔排序
>>希尔排序又称为`缩小增量`排序,把整个列表分成多个$L[i,i+d,i+2d...i+kd]$
这样的列表，每个进行直接插入排序。每一轮不断缩小d的值,直到全部有序(最后的d是1)。

实例代码:
```
#include<iostream>
#include<vector>
#include<string>
#include<ctime>
using namespace std;

void shellSort(vector<int>& array);//其实就相当于直接插入排序
int main()
{
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	vector<int> array = {5, 1, 9, 3, 7, 4, 8, 6, 2 };
	shellSort(array);
	endTime = clock();//计时结束
	cout << "总计时长" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	return 0;
}
void shellSort(vector<int>& array)
{
	int increment = array.size();
	do
	{
		increment = increment / 3 + 1;
		for (int i = increment; i<array.size(); i++)
		{
			int temp = array[i];
			int j = i - increment;
			while (j>=0 && temp<array[j])
			{
				array[j + increment] = array[j];
				j -= increment;
			}
			array[j + increment] = temp;
		}
	} while (increment>1);
}
```
总结分析:

|名称|时间|最好|最坏|空间|稳定|备注|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|希尔排序||$O(n^{1.3})$|$O(n^2)$|$O(1)$|否| |

#选择类排序
##简单选择排序
>>前面已经有序,后面选择最小的与前面交换,从有序序列尾不断增长有序序列

实例代码:
```
#include<iostream>
#include<vector>
#include<string>
#include<ctime>
using namespace std;

void easySelectSort(vector<int>& array);//简单选择排序(适用于数组，链表涉及到交换)
int main()
{
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	vector<int> array = {5, 1, 9, 3, 7, 4, 8, 6, 2 };
	easySelectSort(array);
	endTime = clock();//计时结束
	cout << "总计时长" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	return 0;
}
```
总结分析:

|名称|时间|最好|最坏|空间|稳定|备注|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|简单选择排序|$O(n^2)$|$O(n^2)$|$O(n^2)$|$O(1)$|否| |

##堆排序
>>最好索引从1开始!!!!设当前节点是号是n,当索引从1开始时2*n正处于子树中,而索引从0开始时不然。

实例代码:
```
#include<iostream>
#include<vector>
#include<string>
#include<ctime>
using namespace std;

void heapSort(vector<int>& array);//堆排序必须从1开始，因为若是从0开始当前下标乘以2后的位置就不是当前节点的孩子节点了
void heapAdjust(vector<int>& array, int start, int end);
int main()
{
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	vector<int> array = {5, 1, 9, 3, 7, 4, 8, 6, 2 };
	heapSort(array);
	endTime = clock();//计时结束
	cout << "总计时长" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	return 0;
}
void heapSort(vector<int>& array)
{
	for (int i = (array.size()-1)/2; i >=1; i--)
	{
		heapAdjust(array, i, array.size()-1);
	}
	for (int i = array.size()-1; i > 1; i--)//之所以是>1而不是>=1是因为只剩一个没必要了
	{
		swap(array[1],array[i]);
		heapAdjust(array,1,i-1);
	}
}
void heapAdjust(vector<int>& array,int start,int end)
{
	int temp = array[start];
	for (int i = start*2; i <= end ; i*=2)
	{
		if (i+1<=end && array[i]<array[i+1])//判断+1后是否越界
		{
			i++;
		}
		if (temp>=array[i])
		{
			break;//注意！！！！！无需continue,因为开始建堆的时候就是从下向上调整的，上面不变动下面就肯定不需要变动
		}
		array[start] = array[i];
		start = i;
	}
	array[start] = temp;
}
```
分析总结:

|名称|时间|最好|最坏|空间|稳定|备注|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|堆排序|$O(nlogn)$|$O(nlogn)$|$O(nlogn)$|$O(1)$|否| |

#归并排序(二路)
##由上向下-递归
>>归并排序的形式是一颗二叉树，遍历的次数就是二叉树的深度$O(logn)$,一共n个数

```
递归:
#include<iostream>
#include<vector>
#include<string>
#include<ctime>
using namespace std;

void mergeSort(vector<int>& array, int left, int right);//归并排序
void merge(vector<int>& array, int left, int mid, int right);
int main()
{
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	vector<int> array = {5, 1, 9, 3, 7, 4, 8, 6, 2 };
	mergeSort(array,0,array.size()-1);
	endTime = clock();//计时结束
	cout << "总计时长" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	return 0;
}
void mergeSort(vector<int>& array,int left,int right)
{
	if (left<right)
	{
		int mid = (left + right) / 2;
		mergeSort(array,left,mid);
		mergeSort(array, mid+1, right);
		merge(array,left,mid,right);
	}
}
void merge(vector<int>& array,int left,int mid,int right)
{
	int i = left;
	int j = mid + 1;
	vector<int> temp;
	while (i<=mid && j<=right)
	{
		if (array[i]<array[j])
		{
			temp.push_back(array[i++]);
		}
		else
		{
			temp.push_back(array[j++]);
		}
	}
	while (i<=mid)
	{
		temp.push_back(array[i++]);
	}
	while (j<=right)
	{
		temp.push_back(array[j++]);
	}
	for (int i = 0; i < temp.size(); i++)
	{
		array[left+i] = temp[i];
	}
}
非递归:
#include<iostream>
#include<vector>
#include<string>
#include<ctime>
using namespace std;
void merge(vector<int>& array, int left, int mid, int right);
void merge_sort_down2up(vector<int> &array);
void merge_groups(vector<int>& array, int gap);
int main()
{
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	vector<int> array = {5, 1, 9, 3, 7, 4, 8, 6, 2 };
	merge_sort_down2up(array);
	endTime = clock();//计时结束
	cout << "总计时长" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	return 0;
}
void merge(vector<int>& array,int left,int mid,int right)
{
	int i = left;
	int j = mid + 1;
	vector<int> temp;
	while (i<=mid && j<=right)
	{
		if (array[i]<array[j])
		{
			temp.push_back(array[i++]);
		}
		else
		{
			temp.push_back(array[j++]);
		}
	}
	while (i<=mid)
	{
		temp.push_back(array[i++]);
	}
	while (j<=right)
	{
		temp.push_back(array[j++]);
	}
	for (int i = 0; i < temp.size(); i++)
	{
		array[left+i] = temp[i];
	}
}
void merge_sort_down2up(vector<int>& array) 
{
	for (int i = 1; i < array.size(); i = i * 2)
	{
		merge_groups(array, i);
	}
}
void merge_groups(vector<int>& array, int gap)
{
	int twolen = 2 * gap;
	int i;
	for (i = 0; i + twolen - 1 < array.size(); i += twolen) {
		int start = i;
		int mid = i + gap - 1;
		int end = i + twolen - 1;
		merge(array, start, mid, end);
	}
	// 最后还有一个gap
	if (i + gap - 1 < array.size() - 1) {
		merge(array, i, i + gap - 1, array.size() - 1);
	}
}
```

分析总结:

|名称|时间|最好|最坏|空间|稳定|备注|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|归并排序|$O(nlogn)$|$O(nlogn)$|$O(nlogn)$|$O(n)$|是| |