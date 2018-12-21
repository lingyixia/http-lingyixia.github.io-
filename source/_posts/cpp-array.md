---
layout: post
title: C++数组长度
category: 基础
tags: [C++,坑]
description: C++中获取数组长度的坑
---

>记录函数内获取数组长度的坑

看代码:
```
#include<iostream>
using namespace std;

void fun(int* array)
{
	cout << array << endl;
	cout << sizeof(array);
}

int main()
{
	int array[] = { 1,2,3,4,5,6 };
	cout << array << endl;
	cout << sizeof(array) << endl;
	fun(array);
	return 0;
}
结果：
00DEFCA0
24
00DEFCA0
4
```
可以看出，虽然函数体内外的array所指的地址相同，sizeof后并不一样，前者是实际数组的大小，后者是纯指针的大小（编译器决定），也就是说，当数组传到函数内后，就意味着它是一个纯指针，不再有数组的意义了，要想在函数内获取数组长度，只能以参数的形式传入。
总之，想在函数内获取数组长度就用vector吧！！！