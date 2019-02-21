---
layout: post
title: 各种熵
category: 机器学习
tags: [熵]
date: 2018-12-26 13:11:08
description: 各种熵总结
---

>熵(信息熵)、交叉熵和相对熵(KL散度)、条件熵

### 熵(信息熵)

>信息熵有两个含义:1.系统包含的信息量的期望 2.定量描述该系统所需的编码长度的期望

#### 公式推导

##### 定性推导:

设$h(x)$为$x$包含的**信息量**,如果我们有俩个**不相关**的事件$x$和$y$,那么我们观察到的俩个事件同时发生时获得的**信息量**应该等于观察到的事件各自发生时获得的**信息量**之和,即:$h(x,y) = h(x) + h(y)$,由于$x$,$y$是俩个不相关的事件，则满足$p(x,y) = p(x) \times p(y)$.那么要想让$h(x,y) = h(x) + h(y)$,$h(x)$就只能是$\log p(x)$,为了让信息量为非负,我们在其前面加负号，得到信息量公式:

$$
h(x) = -\log p(x)
$$

##### 定量推导:
[参考该博客](https://blog.csdn.net/stpeace/article/details/79052689)
这个很牛逼！！！！！开始推导：
首先说明信息量的定义，谨记x是个概率值，不是事件:

1. 信息量是概率的递减函数，记为$f(x)$,$x\in[0,1]$  
2. $f(1)=0,f(0)=+∞$  
3. 独立事件(概率之积)的信息量等于各自信息量之和:$f  (x_1 \times x_2)=f(x_1)+f(x_2),x_1,x_2\in[0,1]$  

$$
\begin{align}
f(x)^\prime &= \lim_{\Delta x \to 0}\frac{f(x+\Delta x)-f(x)}{\Delta x} \\
&= \lim_{\Delta x \to 0}\frac{f(\frac{x+\Delta x}{x} \times x)-f(x)}{\Delta x} \\
&= \lim_{\Delta x \to 0}\frac{f(x+\Delta x)+f(x)-f(x)}{\Delta x} \\
&=\lim_{\Delta x \to 0}\frac{f(\frac{x+\Delta x}{x})}{\Delta x} \\
&= \frac{1}{x}\lim_{\Delta x \to 0}\frac{f(1+\frac{\Delta x}{x})}{\frac{\Delta x}{x}} \\
&=\frac{1}{x}f(1)^\prime
\end{align}
$$

积分得:

$$
f(x) = f(1)^\prime\ln|x|+C \qquad x\in[0,1]
$$

令$x=1$,得$C=0$,故

$$
\begin{align}
f(x) &= f(1)^\prime\ln x \\
     &=f(1)^\prime\frac{\log_ax}{\log_ae} \\
     &= \frac{f(1)^\prime}{\log_ae} ] \times \log_ax
\end{align}
$$

而$\frac{f(1)^\prime}{\log_ae}$是个常数，令为1，则

$$
\begin{align}
f(x) &= \log_ax \\
     &= -\log_a\frac{1}{x}
\end{align}
$$

证毕!!!  

##### 编码推导:
[参考此处](https://blog.csdn.net/AckClinkz/article/details/78740427)
首先需要知道的是Karft不等式:$\sum r^{-l_i}\le1$,其中$r$是进制数，一般取二进制,$l_i$表示第$i$个信息的码长.问题可以转化为:  

$$
\min_{l_i}\sum p_il_i \\
s.t.\quad \sum r^{-l_i}\le1
$$

由拉格朗日乘数法:  

$$
L(l_i,\lambda)=\sum p_il_i+\lambda(\sum r^{-l_i}-1)
$$

根据拉格朗日法则求极值得:   

$$
\begin{cases}
p_i-\lambda \times r^{-l_i}\ln r =0 \\
\sum r^{-l_i}-1=0 \\
\end{cases}
$$

由上面公式得:  

$$
\begin{cases}
r^{-l_i}= \frac{p_i}{\lambda \times \ln r} \\
\sum r^{-l_i}=1 \\
\end{cases}
$$

一式带入二式得:$\sum r^{-l_i} = \sum \frac{p_i}{\lambda \times \ln r} = \frac{1}{\lambda \times \ln r} = 1$  

因此:$\lambda = \frac{1}{\ln r}$

最终得:$l_i = \log r \frac{\lambda \times \ln r}{p_i}=\log r\frac{1}{p_i}$，正好是熵!!!


则公式信息熵: 

$$
H(x) = -p(x)\log p(x)
$$

自然是系统信息量的期望，或称为编码长度的期望.

### 交叉熵和相对熵(KL散度)
现在有两个分布，真实分布p和非真实分布q，我们的样本来自真实分布p,按照真实分布p来编码样本所需的编码长度的期望为(**信息熵**):  

$$
H(p) = \sum -p\log p
$$

按照不真实分布q来编码样本所需的编码长度的期望为(**交叉熵**):   

$$
H(p,q)= \sum -p\log q
$$

注意:$H(p,q)≠H(q,p)!!!$而**KL散度(相对熵)**为:

$$
H(p||q)=H(p,q)-H(p)
$$

它表示两个分布的差异，差异越大，相对熵越大。

### 条件熵
条件熵针对得是某个特征范围内来计算，而不是整个样本，即熵是整个样本的不确定性，而条件熵是特定特征条件下样本的不确定性。[案例见此](https://blog.csdn.net/xwd18280820053/article/details/70739368),公式表达为:

$$
\begin{align}
H(Y|X) &=\sum_{x\in X}p(X)H(Y|X=x) \\
       &=\sum _{x,y}-p(x,y)\log p(y|x)
\end{align}
$$

X表示总体样本的某个特征，条件熵和其他熵最大的不同就是条件熵和特征有关，其他熵只和Label有关，和特征无关。

### 联合熵
表示X和Y是同一个分布中的两个特征，没有特征和Label之分.

$$
H(X,Y) = \sum_{x,y}-p(x,y)\log p(x,y)
$$

联合熵和条件熵的关系:  

$$
\begin{align}
H(X,Y) - H(Y|X) &= -\sum_{x,y}p(x,y)\log p(x,y) + \sum_{x,y}p(x,y)\log p(y|x) \\
&= -\sum_{x,y}p(x,y)\log p(x) \\
&=-\sum_xp(x)\log p(x) \\
&=H(X)
\end{align}
$$

注意，要和条件熵的符号区分。

### 互信息(信息增益)

$$
\begin{align}
I(X,Y) &= H(X)-H(X|Y) \\
       &= H(Y)-H(Y|X) \\
       &= H(X)+H(Y) - H(X,Y) \\
       &= H(X,Y) - H(X|Y) -H(Y|X)
\end{align}
$$
