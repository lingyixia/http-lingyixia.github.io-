---
title: C++初始化列表知识
date: 2019-06-02 16:58:52
category: 基础
tags: [C++]
---
>使用初始化列表初始化成员变量的时候要注意,初始化的顺序和初始化列表无关,顺序只和变量声明的顺序有关

eg:
```
#include <iostream>
using namespace std;

class A
{
private:
	int n1;


	int n2;
publ#include <iostream>
using namespace std;

class A
{
private:
	int n2;
	int n1;
public:
	A() :n2(0), n1(n2 + 2) {}
	void Print()
	{
		cout << n1 << " " << n2 << endl;
	}
};

int main()
{
	A a;
	a.Print();
	return 0;
}ic:
	A() :n2(0), n1(n2 + 2) {}
	void Print()
	{
		cout << n1 << " " << n2 << endl;
	}
};

int main()
{
	A a;
	a.Print();
	return 0;
}
```

>>结果是:一个随机数,0.因为变量声明的顺序是先n1后n2,因此初始化列表中先n1初始化,此时n2尚未初始化,因此只能是一个随机数

```
#include <iostream>
using namespace std;

class A
{
private:
	int n2;
	int n1;
public:
	A() :n2(0), n1(n2 + 2) {}
	void Print()
	{
		cout << n1 << " " << n2 << endl;
	}
};

int main()
{
	A a;
	a.Print();
	return 0;
}
```
>>输出:2,0,因为n2先于n1声明,因此初始化列表中先初始化n2,后n1