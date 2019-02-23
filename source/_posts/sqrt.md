---
title: 开平方
date: 2019-01-22 09:29:55
category: 算法
tags: [二分法,牛顿法]
---

>二分法和牛顿法求一个非负数开平方

#二分法

```
double sqrtBinary(double x)
{
	clock_t startTime, endTime;
	startTime = clock();
	double min=0.0;
	double max=x/2+1;
	double p = 1e-5;
	double mid;
	int interIndex = 0;
	while ((max-min)>=p)
	{
		interIndex++;
		mid = (min + max) / 2;
		if (mid*mid>x)
		{
			max = mid;
		}
		else if (mid*mid<x)
		{
			min = mid;
		}
		else
		{
			return mid;
		}
	}
	endTime = clock();
	cout <<"迭代次数:"<<interIndex<<"计算时间:"<< (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	return mid;
}
```
#牛顿法
```
double sqrtNewTon(double n) 
{
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	if (n == 0)
	{
		return 0;
	}
	double last = 0.0;
	double current = 1.0;
	double p = 1e-5;
	int interIndex = 0;
	while (fabs(current - last) >= p)
	{
		interIndex++;
		last = current;
		current = (last + n / last) / 2;
	}
	endTime = clock();//计时开始
	cout <<"迭代次数:"<<interIndex<<"计算时间:"<< (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	return current;
}
```

牛顿法解释:
![](/img/newton.jpg)  
对于求$x^2=n$的$x$,首先令$f(x)=x^2-n$,任意取$x_0$,则点为$(x_0,f(x_0))$,则该点的切线方程为:
$$
f(x)=f(x_0)+f(x_0)^{\prime}(x-x_0)
$$
即:
$$
\begin{align}
f(x)&=x_0^2-n+(2*x_0)(x-x_0) \\
&=2x_0x-(x_0^2+n)
\end{align}
$$
令$f(x)=0$,得$x=\frac{(x_0+\frac{n}{x_0})}{2}$
即:
$$
x_{i+1}=\frac{(x_i+\frac{n}{x_i})}{2}
$$
其实就是**梯度下降**的前身。
[详细见此](https://www.zhihu.com/question/20690553)