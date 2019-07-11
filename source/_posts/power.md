---
title: 数值的整数次幂
date: 2019-03-15 16:17:38
category: 算法
tags: [Tip]
---
>给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

```
double unsignedPower(double base, unsigned int exponent)
{
	if (exponent == 0) return 1;
	double result = unsignedPower(base, exponent >> 1);
	result *= result;
	if ((exponent & 1) == 1)//无符号数判断奇数偶数,切记前面加括号,因为"=="比"&"优先级高
	{
		result *= base;
	}
	return result;
}
double Power(double base, int exponent)
{
	if (base == 0.0 && exponent <= 0)
	{
		cout << "无效输入" << endl;
	}
	unsigned int unsignedExponent = abs(exponent);
	double result = unsignedPower(base, unsignedExponent);
	if (exponent >= 0)
	{
		return result;
	}
	return 1.0 / result;
}
```
>>两个点:一，除2用右移计算，如:9>>1=4,-9>>1=-5 二，判断奇数偶数和1作&运算，其实就是判断其二进制最后一位是不是1。