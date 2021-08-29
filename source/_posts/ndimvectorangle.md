---
title: n维空间任意两向量夹角分布
date: 2021-08-26 14:29:23
category: 数学
tags: []
---
#命题
本文要证明的是高**n维空间中，任意两个向量都几乎垂直**，注意：不是两个向量垂直的概率大，而是"几乎"垂直，即夹角接近$\frac{π}{2}$。基本思路就是考虑两个随机向量的夹角$\theta$分布，然后求导得到概率密度，就可以看出在$\theta$在哪个范围内最大。
#命题重定义
随机两个向量不好求，我们可以先固定一个，让另一个随机即可，假设固定向量为：
$$x=(1,0,...,0) \tag{1}$$
随机向量为：
$$y=(y_1,y_2,...,y_n) \tag{2}$$
现在我们把原命题重新定义为**n维单位超球面上，任意一个点与原点组成的单位向量和$(1,0,0,...,0)$向量都几乎垂直**
直接计算可以得到:
$$cos \langle x,y \rangle=\frac{x_1}{\sqrt(x_1^2+x_2^2+...x_n^2)} \tag{3}$$
现在要求的就是公式(3)的概率分布和密度，到这里还是一筹莫展。
#球坐标系
将$y$直角坐标转为球坐标系后为：
$$
\left\{
    \begin{aligned}
    y_1 & =  cos(\varphi_1) \\
    y_2 & =  sin(\varphi_1)cos(\varphi_2) \\
    y_3 &= sin(\varphi_1)sin(\varphi_2) cos(\varphi_3) \\
    .
    .
    .\\
    y_{n-1} &= sin(\varphi_1)sin(\varphi_2)...sina(\varphi_{n-2}) cos(\varphi_{n-1}) \\
    y_{n} &= sin(\varphi_1)sin(\varphi_2)...sina(\varphi_{n-2}) sin(\varphi_{n-1})
    \end{aligned}
\right. \tag{4}
$$
其中，$\varphi\_{n-1} \in[0,2π]$,  $\varphi\_{0...n-2} \in[0,π]$
此时,公式(3)中$cos \langle x,y \rangle$恰好等于$\varphi_1$,即两者之间的角度就是$\varphi_1$
$$
P_n(\varphi_1<\theta)=\frac{n维超球面上\varphi_1<\theta 部分面积(可以想像为三维球体的环带)}{n维超球体表面积} \tag{5}
$$
n维超球面上的积分微元是$\sin^{n−2}(\varphi\_1)\sin^{n−3}(\varphi\_2)⋯sin(\varphi\_{n−2})d\varphi1d\varphi2⋯d\varphi\_{n−2}d\varphi\_{n−1}$ 
因此n维球面积积分为:
$$
\begin{align*}
S_n&=\int_0^{2π}d\varphi_{n−1}\int_0^πsin(\varphi_{n−2})d\varphi_{n−2}...\int_0^π\sin^{n−3}(\varphi_2)d\varphi_2\int_0^π\sin^{n−2}(\varphi_1)d\varphi_1 \tag{6}
\end{align*}
$$
故:
$$
\begin{align*}
P_n(\varphi_1<\theta) &= \frac{n维超球面上\varphi_1<\theta 部分面积}{n维超球体表面积} \\
&=\frac{\int_0^{2π}\int_0^π...\int_0^π \int_0^\theta \sin^{n−2}(\varphi_1)\sin^{n−3}(\varphi_2)⋯sin(\varphi_{n−2})d\varphi1d\varphi2⋯d\varphi_{n−2}d\varphi_{n−1}}{\int_0^{2π}\int_0^π...\int_0^π \int_0^π\sin^{n−2}(\varphi_1)\sin^{n−3}(\varphi_2)⋯sin(\varphi_{n−2})d\varphi_1d\varphi_2⋯d\varphi_{n−2}d\varphi_{n−1}} \\
&=\frac{\int_0^{2π}d\varphi_{n−1}\int_0^πsin(\varphi_{n−2})d\varphi_{n−2}...\int_0^π\sin^{n−3}(\varphi_2)d\varphi_2\int_0^\theta\sin^{n−2}(\varphi_1)d\varphi_1}{\int_0^{2π}d\varphi_{n−1}\int_0^πsin(\varphi_{n−2})d\varphi_{n−2}...\int_0^π\sin^{n−3}(\varphi_2)d\varphi_2\int_0^π\sin^{n−2}(\varphi_1)d\varphi_1} \\
&=\frac{(n−1)维单位超球的表面积 \times \int_0^\theta\sin^{n−2}(\varphi_1)d\varphi_1}{n维单位超球的表面积} \\
&=\frac{\Gamma(\frac{n}{2})}{\Gamma(\frac{n-1}{2})\times \sqrt(π)} \times \int_0^\theta\sin^{n−2}(\varphi_1)d\varphi_1 \tag{7}
\end{align*}
$$
>>小插曲:
n维球体积:$V\_n=\frac{\pi^{\frac{n}{2}}}{\Gamma(n/2)}r^{n-1}$,n维球面积$S\_n=\frac{2\pi^{\frac{n}{2}}}{\Gamma(\frac{n}{2})}r^{n-1}=V\_n'$

因此,  概率密度为:
$$
p_n(\theta)=P_n(\varphi_1<\theta)'=\frac{\Gamma(\frac{n}{2})}{\Gamma(\frac{n-1}{2})\times \sqrt(π)} \times \sin^{n−2} \tag{8}\theta
$$
#密度函数图像
```
#!/usr/bin/env python
# coding=utf-8

import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def fun(n, x):
    return (math.gamma(n / 2) * math.pow(math.sin(x), n - 2)) / (math.sqrt(math.pi) * math.gamma((n - 1) / 2))


if __name__ == '__main__':
    datas_dict = dict()
    xs = np.arange(0, 3, 0.01)
    ys = list()
    for n in [2, 3, 5, 10, 50, 100]:
        for x in xs:
            ys.append(fun(n, x))
        datas_dict[str(n)] = ys.copy()
        ys.clear()
    data = pd.DataFrame(datas_dict, index=list(xs))
    sns.lineplot(data=data)
    plt.show()

```
![](/img/nvector.jpeg)