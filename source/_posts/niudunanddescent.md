---
title: 牛顿法和梯度下降法
date: 2019-03-13 21:32:45
category: 深度学习
tags:
---

牛顿法和梯度下降对比
<!--more-->
#牛顿法
首先需要确定,牛顿法是为了求解函数值为零的时候变量的取值问题的，具体地，当要求解 f(θ)=0时，如果 f可导，那么可以通过迭代公式
初始化参数$\theta_0$,则$\theta_0$对应在函数$f(\theta)$上是$(\theta_0,f(\theta_0))$,导数为$f(\theta_0)\prime$,则过该点的直线为:
$$
f(\theta)=f(\theta_0)+f(\theta_0)\prime(\theta-\theta_0) \tag{1.1}
$$
该直线与$\theta$轴的交点为:
$$
\theta=\theta_0-\frac{f(\theta_0)}{f(\theta_0)\prime} \tag{1.2}
$$
即:
$$
\theta_{n+1}=\theta_n-\frac{f(\theta_n)}{f(\theta_n)\prime} \tag{1.3}
$$
这就是**牛顿法**的参数更新公式.
在神经网络更新参数的时候:
但是牛顿法中我们的更新目标不是要让$f(x)=0$,而是要让$f(x)\prime = 0$,因此,更新公式应该是:
$$
\theta_{n+1}=\theta_n-\frac{f(\theta_n)\prime}{f(\theta_n)\prime \prime} \tag{1.4}
$$

#梯度下降法
$f(\theta;x)$是参数为$\theta$的损失函数,以下就用$f(\theta)$代替。
由泰勒公式:
$$
f(\theta+\Delta \theta)≈f(\theta)+f(\theta)\prime \Delta \theta \tag{2.1}
$$
我们迭代的目标是让$f(\theta)$在更新过程中变小,因此,令$\Delta \theta = \eta f(\theta)\prime$,其中$\eta > 0$:
$$
f(\theta-\eta f(\theta)\prime) ≈ f(x)+\eta f(x)\prime ^2 \tag{2.2}
$$
也就是说式(2.2)中满足$f(\theta)>f(\theta-\eta f(\theta)\prime)$(忽略约等于),这意味着当我们按照下面的公式更新$\theta$的时候:
$$
\theta_n=\theta_{n-1}-\eta f(\theta)\prime \tag{2.3}
$$
就能不断的将目标函数$f(\theta)$缩小。