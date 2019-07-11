---
title: memset讲解
date: 2019-02-10 17:11:57
category: 基础
tags: [C++]
---
>原型为:void *memset(void *s, int ch, size_t n).解释:将以s为起始位置,n字节大小的区域,按字节填充,**每个字节填充ch**。注意,ch指的是Ascii码的第ch个,也就是说它最大也就是256.一般用来填充0

eg1:
```
int main()
{
	int* number = new int[10];
	memset(number,0,10);
    cout<<number[0]<<endl;
	return 0;
}
输出:0
解释:此时的ch是0,也就是ascii码的第0个,也就是00000000,故number全填充为0
```

eg2:
```
int main()
{
	int* number = new int[10];
	memset(number,1,10);
    cout<<number[0]<<endl;
	return 0;
}
输出:16843009
解释:此时ch是1,也就是ascii码的第1个,也就是00000001,故number[0]是:00000001000000010000000100000001,计算即可得到
```