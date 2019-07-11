---
title: 进制
date: 2019-01-14 16:30:32
category: 基础
tags: [C++]
---
C++输出数字二进制、十进制、八进制、十六进制
<!--more-->
```
#include<iostream>
#include <bitset>
using namespace std;

int main() {
	int num = -42;
	cout << bitset<sizeof(num) * 8>(num) << endl;
	cout << num << endl; //默认十进制输出
	cout << hex << num << endl;
	cout << oct << num << endl;

	return 0;
}
输出:
11111111111111111111111111010110
-42
ffffffd6
37777777726
```
