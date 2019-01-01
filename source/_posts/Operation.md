---
layout: post
title: C++操作符重载
category: 基础
tags: [C++]
description: C++中成员函数和非成员函数两种操作符重载
---
>以平面点为例记录C++中操作符重载写法

#重载规则
1. 除了类属关系运算符 " **.** " 、成员指针运算符 " **.*** " 、作用域运算符 " **::** " 、**sizeof运算符**和三目运算符 " **?:** " 以外，C++ 中的所有运算符都可以重载。
2. C++不允许用户自己定义新的运算符，只能对已有的C++运算符进行重载。
3. 重载不能改变运算符运算对象（即操作数）的个数。(双目运算符重载还是双目,单目运算符重载还是单目)
4. 重载之后的运算符不能改变运算符的优先级和结合性(如复制运算符”=“是右结合性（自右至左），重载后仍为右结合性)，也不能改变运算符操作数的个数及语法结构。
5. 重载运算符的函数不能有默认的参数
6. 重载的运算符必须和用户定义的自定义类型的对象一起使用，其参数至少应有一个是类对象（或类对象的引用）,也就是说，参数不能全部是C++的标准类型.
8. 用于类对象的运算符一般必须重载，但有两个例外，运算符”=“和运算符”&“不必用户重载。
7. 运算符重载函数可以是类的成员函数，也可以是类的**友元函数**(可以访问类的私有成员)，还可以是既非类的成员函数也不是友元函数的普通函数。
#实例代码
```
#include<iostream>
using namespace std;
struct Point
{
	int x, y;
	Point() :x(0), y(0) {}
	Point(int _x, int _y) :x(_x), y(_y) {}
	Point operator+(Point& p)
	{
		Point result;
		result.x = this->x + p.x;
		result.y = this->y + p.y;
		return result;
	}
	friend int operator*(Point &p1, Point &p2);  //实现内积
	Point operator++(int pos)//有参数后置
	{
		return Point(this->x++, this->y++);
	}
	Point operator++()//无参数前置
	{
		return Point(++this->x, ++this->y);
	}
};
int operator*(Point &p1, Point &p2)
{
	return (p1.x*p2.x) + (p1.y*p2.y);
}
输出:
20
p3:(3,9)
p4:1
p4:(2,3)
p4:2
```