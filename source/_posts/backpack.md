---
layout: post
title: 背包问题
category: 算法
tags: [递归,背包,动态规划]
description: 三种背包问题
---

>0-1背包、完全背包、多重背包问题简记

#0-1背包问题
##常规写法
递推公式为:`values[i][j] = max(values[i - 1][j], values[i-1][j - woods[i-1].volume] + woods[i-1].value);`
```
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
int items[5] = {0,0,0,0,0};
struct Woods
{
	int volume;
	int value;
};
void backPack(vector<Woods> woods, int bag);
void findItems(int** values, vector<Woods> woods, int i, int j);
int main()
{
	vector<Woods> woods = {Woods{2,3},Woods{3,4},Woods{4,5},Woods{5,6} };
	backPack(woods,8);
	return 0;
}
void backPack(vector<Woods> woods, int bag)
{
	int** values = new int*[woods.size()+1];
	for (int i = 0; i <= woods.size(); i++)
	{
		values[i] = new int[bag+1]();
	}
	for (int i = 1; i <= woods.size(); i++)
	{
		for (int j = 1; j <= bag; j++)
		{
			if (j < woods[i-1].volume)
			{
				values[i][j] = values[i - 1][j];
			}
			else
			{
				values[i][j] = max(values[i - 1][j], values[i-1][j - woods[i-1].volume] + woods[i-1].value);
			}
		}
	}
	findItems(values, woods, 4, 8);
	for (int i = 0; i <= woods.size(); i++)
	{
		delete [] values[i];
	}
	delete values;
}
void findItems(int** values, vector<Woods> woods,int i ,int j)
{
	if (i>0)
	{
		if (values[i][j] == values[i-1][j])
		{
			items[i] = 0;
			findItems(values, woods,i-1,j);
		}
		else
		{
			items[i] = 1;
			findItems(values, woods,i-1,j-woods[i-1].volume);
		}
	}
}
```
上诉方法使用二维数组保存中间值，比较消耗空间，而且我们可以看出，对于每一步要求的`values[i][j]`而言，只依赖于`values[i-1][:]`,即前一行，因此我们只需要一维数组即可，循环保存前一行数据。但是这样貌似就无法找物品组成了，回头在想。
##空间优化写法
```
void backPack(vector<Woods> woods, int bag)
{
	int* values = new int[bag + 1]();
	for (int i = 1; i <= woods.size(); i++)
	{
		for (int j = bag; j >= 0; j--)//倒序，保证在改完某个数据之前，它依赖的前面的数据还没改
		{
			if (j - woods[i-1].volume >= 0)
			{
				values[j] = max(values[j], values[j - woods[i-1].volume] + woods[i-1].value);
			}
		}
	}
}
```
##再次改善
```
void backPack(vector<Woods> woods, int bag)
{
	int* values = new int[bag + 1]();
	for (int i = 1; i <= woods.size(); i++)
	{
		for (int j = bag; j >= woods[i-1].volume; j--)//倒序，保证更改某个数据之前它依赖的前面的数据未改
		{
			values[j] = max(values[j], values[j - woods[i-1].volume] + woods[i-1].value);
		}
	}
}
```

#完全背包问题
##常规写法
与0-1背包唯一的不同就是递推公式：`values[i][j] = max(values[i - 1][j], values[i][j - woods[i-1].volume] + woods[i-1].value);`不在详细说明

##空间优化写法
```
void backPack(vector<Woods> woods, int bag)
{
	int* values = new int[bag + 1]();
	for (int i = 1; i <= woods.size(); i++)
	{
		for (int j = woods[i-1].volume; j <= bag; j++)//正序
		{
			values[j] = max(values[j], values[j - woods[i-1].volume] + woods[i-1].value);
		}
	}
}

```

#多重背包问题（即每种物品的数量有上限）
##常规写法
在0-1背包的基础上增加数量循环判断,递推公式为:`values[i][j] = max(values[i][j], values[i-1][j - k * woods[i - 1].volume] + k * woods[i - 1].value);`
```
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
struct Woods
{
	int volume;
	int value;
	int num;
};
void backPack(vector<Woods> woods, int bag);
int main()
{
	vector<Woods> woods = { Woods{ 80 ,20 ,4 },Woods{ 40 ,50, 9 },Woods{ 30 ,50, 7 },Woods{ 40 ,30 ,6 },Woods{ 20 ,20 ,1 } };
	backPack(woods, 1000);
	return 0;
}
void backPack(vector<Woods> woods, int bag)
{
	int** values = new int*[woods.size() + 1];
	for (int i = 0; i <= woods.size(); i++)
	{
		values[i] = new int[bag + 1]();
	}
	for (int i = 1; i <= woods.size(); i++)
	{
		for (int j = 1; j <= bag; j++)
		{
			values[i][j] = values[i - 1][j];
			for (int k = 1; k <= woods[i - 1].num; k++)
			{
				if (j < k * woods[i - 1].volume)
				{
					break;
				}
				values[i][j] = max(values[i][j], values[i-1][j - k * woods[i - 1].volume] + k * woods[i - 1].value);
			}
		}
	}
	for (int i = 0; i <= woods.size(); i++)
	{
		delete[] values[i];
	}
	delete values;
}
```
##空间优化写法
```
void backPack(vector<Woods> woods, int bag)
{
	int* values = new int[bag + 1]();
	for (int i = 1; i <= woods.size(); i++)
	{
		for (int j = bag; j >= 0; j--)
		{
			for (int k = 1; k <= woods[i - 1].num; k++)
			{
				if (j < k * woods[i - 1].volume)
				{
					break;
				}
				values[j] = max(values[j], values[j - k * woods[i - 1].volume] + k * woods[i - 1].value);
			}
		}
	}
	delete[] values;
}
```

二进制写法先不想了
