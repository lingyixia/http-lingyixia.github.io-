---
title: 常用分布
date: 2019-06-23 15:03:04
category: 数学
tags: [概率论]
---

常用分布若干知识点记录
<!--more-->

#伯努利分布
>>进行一次实验,实验的结果只有两种:成功的概率是$p$,则失败的概率是$1-p$。eg:一个硬币抛一次人结果。
$$
P(X=1)=p \\
P(X=0)=1-p
$$
期望:$E(X)=P$
方差:$D(X)=P \times (1-P)$

#二项分布
>>n次伯努利实验,成功k次。g:一个硬币抛n次，k次正面朝上。
$$
P(X=k)=C_n^kp^{k}(1-p)^{n-k}
$$
期望:
$$
\begin{align}
E(X) &=\sum_{k=0}^nk C_n^kp^{k}(1-p)^{n-k} \\
&=\sum_{k=0}^nk \frac{n!}{k!(n-k)!} p^{k}(1-p)^{n-k} \\
&=n\sum_{k=0}^n \frac{(n-1)!}{(k-1)!(n-k)!} p^{k}(1-p)^{n-k} \\
&= np\sum_{k=0}^n C_{n-1}^{k-1}p^{k-1}(1-p)^{(n-1)-(k-1)} \\
&=np(p+(1-p))^{n-1} \\
&=np
\end{align}
$$
$$
D(X)=np(1-p)
$$

>>过两天在证明

#几何分布
>>考虑伯努利试验，每次成功的概率为$p$,$0<p<1$,重复试验直到试验首次成功为止。令X表示需要试验的次数，那么:
$$
P(X=n)=(1−p)^{n−1}p
$$
* $E(X)=\frac{1}{p}$
* $D(X)=\frac{1-p}{p^2}$
[参考](https://zlearning.netlify.com/math/probability/geometry-distribution.html)

#补充:
$$
\begin{align}
D(X) &= E(X-E(X)^2) \\
&=E(X^2-2XE(X)+E(X)^2) \\
&=E(X^2)-2E(x)^2+E(X)^2 \\
&=E(X^2)-E(X)^2
\end{align}
$$