---
layout: post
title: C++参数传递方式
category: 基础
tags: [C++]
description: C++三种参数传递方式
---

>C++函数的三种传递方式为：值传递、指针传递和引用传递

### 值传递(略)
### 指针传递(实质也是值传递)
```
void fun(int *x)
{
    *x += 5; //修改的是指针x指向的内存单元值
}
void main(void)
{
    int y = 0;
    fun(&y);
cout<<<<\"y = \"<<y<<endl; //y = 5;
}
```
>>实质是传递地址的值，即地址的‘值传递’
### 引用传递
```
void fun(int &x)
{
   x += 5; //修改的是x引用的对象值 &x = y;
}
void main(void)
{
int y = 0;
fun(y);
cout<<<<\"y = \"<<y<<endl; //y = 5;
}
```
>>实质是取别名，&在C++中有两种作用，取别名和取地址（C中只有后者）,=号左边是引用，=号右边是取址。
```
    int a=3;  
    int &b=a;//引用              
    int *p=&a; //取地址
```

引用传递是C++的特性，值传递和指针传递是C语言中本来就有的方式。