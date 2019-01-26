---
layout: post
title: 线性回归和逻辑回归对比
category: 机器学习
tags: [回归,分类,有监督]
description: 线性回归和逻辑回归对比
---

>温故而知新啊，今天复习线性回归和逻辑回归，发现了以前没想过的东西,即为什么逻辑回归要用交叉熵函数。

### 简单对比

1. 最终函数
线性回归:
$$f(x)_{w,b}=wx+b$$
逻辑回归:
$$
P(Y=1|X)=\frac{e^{wx+b}}{1+e^{wx+b}}\\
P(Y=0|X)=\frac{1}{1+e^{wx+b}}
$$
令:
$$f(x)_{w,b}=P(1|X)=\frac{e^{wx+b}}{1+e^{wx+b}}$$
2. 损失函数
线性回归(注意谁减谁):
$$L(w,b)=\frac{1}{2}\sum_{i=1}^N(y_i-f(x_i))^2$$
逻辑回归(注意谁前谁后):
$$L(w,b)=\sum_{i=1}^NC(f(x_i),y_i)$$
3. 梯度计算:
线性回归:
$$\nabla_wL(w,b)=\sum_{i=1}^N(y_i-f(x_i))(-x_i)\\
\nabla_bL(w,b)=\sum_{i=1}^N(y_i-f(x_i))
$$
逻辑回归:
$$\begin{align}
\sum_{i=1}^NC(f(x_i),y_i) &= \sum_{i=1}^N-[y_i\ln(f(x_i))+(1-y_i)\ln(1-f(x_i))]\\
						  &= -\sum_{i=1}^N[y_i\ln(\frac{f(x_i)}{1-f(x_i)})+\ln(1-f(x_i))]\\
						  &= -\sum_{i=1}^N[y_i(wx_i+b)+\ln(1-f(x_i))]\\
						  &= -\sum_{i=1}^N[y_i(wx_i+b)-\ln(1+e^{wx+b})]
\end{align}$$
$$
\nabla_wL(w,b)=-\sum_{i=1}^N(y_i-f(x_i))x_i\\
\nabla_bL(w,b)=-\sum_{i=1}^N(y_i-f(x_i))
$$
至此发现线性回归和逻辑回归的参数偏导公式完全相同，然后梯度上升或下降即可(上升还是下降取决于线性回归谁减谁，逻辑回归交叉熵谁先谁后)。

### 交叉熵含义
对于
$$T=\{(x_1,y_1),(x_2,y_2),...(x_N,y_N)\}$$
要想让得到回归函数$f(x)$最符合要求,只需使**后验概率概率**最大即可:
$$\prod_{i=1}^N[f(x_i)]^{y_i}[1-f(x_i)]^{1-y_i}$$
其中,$y_i$是标签为1的数据，这其实是个似然函数,然后取$\log$:
$$\begin{align}
L(w,b) &= \sum_{i=i}^N[y_i\log(f(x_i))+(1-y_i)\log(1-f(x_i))]\\
       &= \sum_{i=i}^N[y_i(wx_i+b)-\ln(1+e^{wx+b})]
\end{align}$$
发现$L(w,b)=\sum_{i=1}^NC(f(x_i),y_i)$,因此，**交叉熵的含义其实就是后验概率最大化**。