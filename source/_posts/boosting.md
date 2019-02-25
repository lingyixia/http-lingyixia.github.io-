---
title: boosting
date: 2019-02-22 19:04:20
category: 机器学习
tags: [分类,回归]
---
<<<<<<< HEAD
>boosting 是一种特殊的集成学习方法。所有的‘基’分类器都是弱学习器，但通过采用特定的方式迭代，每次根据训练过的学习器的预测效果来更新样本权值,用于新的一轮学习,最终提高联合后的学习效果boosting

#AdaBoost
`AdaBoost`算法中,标签`y`是{-1,1}.
目标函数是:
=======
>boosting 是一种特殊的集成学习方法。所有的‘基’分类器都是弱学习器，但通过采用特定的方式迭代，每次根据训练过的学习器的预测效果来更新样本权值,用于新的一轮学习,最终提高联合后的学习效果boosting。

#AdaBoost

##步骤:
1. 首先，是初始化训练数据的权值分布D1。假设有N个训练样本数据，则每一个训练样本最开始时，都被赋予相同的权值:w1=1/N。
2. 然后，训练弱分类器$G_i$.具体训练过程中是:如果某个训练样本点，被弱分类器$G_i$准确地分类,那么在构造下一个训练集中,它对应的权值要减小;相反,如果某个训练样本点被错误分类,那么它的权值就应该增大。权值更新过的样本集被用于训练下一个分类器，整个训练过程如此迭代地进行下去。
3. 最后，将各个训练得到的弱分类器组合成一个强分类器。各个弱分类器的训练过程结束后，加大分类误差率小的弱分类器的权重，使其在最终的分类函数中起着较大的决定作用，而降低分类误差率大的弱分类器的权重，使其在最终的分类函数中起着较小的决定作用。
换而言之,误差率低的弱分类器在最终分类器中占的权重较大,否则较小。

##流程
1. 每个训练样本初始化相同权值,即$w_i=1/N,得到$:
$$
D_1=(w_{11},w_{12},...,w_{1N}),w_{1i}=\frac{1}{N}
$$
2. 开始迭代$m=1$~$M$
a. 选取当前权值分布下误差率最低的分类器G为第m个基本分类器$G_m$,并计算误差:
$$
e_m=\sum_{t=1}^NP(G_m(x_i)≠y_i)=\sum_{i=1}^Nw_{mi}I(G_m(x_i)≠y_i)
$$
b. 计算分类器权重:
$$
\alpha_m=\frac{1}{2}\ln \frac{1-e_m}{e_m}
$$
此处需要备注:当$e_m<1/2$时,$\alpha_m>0$，且$\alpha_m$随着$e_m$的增大而减少,即分类误差越小该弱分类器权值越大,意味着在最终的全国分类器中该弱分类器的权值大。
c. 更新样本权值分布$D_{t+1}$:
$$
\begin{gather}
D_{m+1}=(w_{m+1,1},w_{m+1,2},...,w_{m+1,N}) \\
w_{m+1,i}=\frac{w_{mi}}{Z_m} e^{-\alpha y_iG_m(x_i)}
\end{gather}
$$
其中,$Z_m$是规范化因子:
$$
Z_m=\sum_i^N w_{mi}e^{-\alpha_m y_i G_m(x_i)}=2 \sqrt{e_m(1-e_m)}(可证)
$$
此处需要备注:可以看出:
$$
w_{m+1,i} = \begin{cases}
\frac{w_{mi}}{Z_m}e^{-alpha_m} &  G_m(x_i) = y_i( G_m(x_i) \times y_i =1) \\
\frac{w_{mi}}{Z_m}e^{alpha_m} &  G_m(x_i) ≠ y_i(G_m(x_i) \times y_i = -1) \\
\end{cases}
$$
意味着某样本分类错误时候该样本权值扩大,即所谓的更加关注该样本.

3. 组合分类器 
$$
f(x) = \sum_{m=1}^M \alpha_m G_m(x_i)
$$
得到最终分类器:
$$
G(x) = sign(f(x))=sign(\sum_{m=1}^M \alpha_m G_m(x_i))
$$
#Boosting Tree(提升树算法)
##模型
采用加法模型,利用前向分布算法,第m步得到的模型为:
$$
f_m(x)=f_{m-1}(x)+T(x;\Theta_m)
$$
其中$T(x;\Theta_m)$表示第m部要得到的决策树,$\Theta_m$表示其参数,并通过经验风险极小化来确定$\Theta_m$:
$$
\hat{\Theta}_m=\arg \min_{\Theta_m} \sum_{i=1}^NL(y_i,f_{m-1}(x_i)+T(x_i;\Theta_m))
$$
##学习过程
$$
\begin{align}
f_0(x)&=0 \\
f_m(x)&=f_{m-1}(x)+T(x;\Theta_m) \\
f_m(x)&=\sum_{m=1}^M T(x;\Theta_m)
\end{align}
$$
当使用平方误差损失函数时:
$$
L(y,f(x))=(y-f(x))^2
$$
即:
$$
\begin{align}
L(y,f_{m-1}(x)+T(x;\Theta_m))&=(y-f_{m-1}(x)-T(x;\Theta_m))^2 \\
&=(r-T(x;\Theta_m))^2
\end{align}
$$
其中$r=y-f_{m-1}(x)$,即第m-1论求得的树还剩下的**残差**,现在第m轮的目标是减少这个**残差**.


#梯度提升
>>主要思想和上诉算法相同,不同点在于使用负梯度(伪残差)代替残差

1. 初始化
$$
f_0(x) = \arg \min_c \sum_{i=1}^N L(y_i,c)
$$
2. 对m=1,2,3...,M
a. 对i=1,2,3...,N,计算伪残差:
$$
r_{mi}=-[\frac{\partial L(y,f(x_i))}{\partial f(x_i)}]_{f(x)=f_{m-1}(x)}
$$
b. 使用该残差列表$(x_i,r_{mi})$计算新的分类器$f_{m}$
c. 计算部长(也就是所谓的学习率)
$$
\gamma _m=\arg \min_{\gamma} \sum_{i=1}^N L(y_i,f_{m-1}-\gamma f_m(x))
$$
d. 更新模型:
$$
f_m(x) = f_{m-1}(x)-\gamma_m f(x)
$$

#GBDT
>>当上诉地图提升使用的是基函数是决策树的时候就是GBDT
#XGboost
>>补充:损失函数：计算的是一个样本的误差代价函数：是整个训练集上所有样本误差的平均目标函数：代价函数 + 正则化项

损失函数为:
$$
Obj=\sum_{i=1}^n l(y_i,\hat{y_i})+\Omega(f_t)
$$
第一项是loss,第二项是正则项:
$$
\Omega(f_t) = \gamma T + \frac{1}{2}\gamma\sum_{j=1}^T w_j^2
$$
其中$w_j^2$是叶子节点j的权重.
由于新生成的树要拟合上次预测的损失,因此有:
$$
\hat{y_i} = \hat{y_i}^{(t-1)} + f_t(x_i)
$$
同时,可以将目标函数改写为:
$$
L^t=\sum_{i=1}^n l(y_i,\hat{y_i}^{(t-1)}+f_t(x_i))+\Omega(f_t)
$$
二阶泰勒展开:
$$
L^t≈ \sum_{i=1}^n[l(y_i,\hat{y}^{(t-1)}+g_if_t(x_i)+\frac{1}{2}f_t^2(x_i))]+\Omega(f_t)
$$
其中$g_i$和$h_i$分别是一阶和二阶导数.可以直接去掉t-1棵树的残差:
$$
L^t≈ \sum_{i=1}^n[g_if_t(x_i)+\frac{1}{2}f_t^2(x_i))]+\Omega(f_t)
$$
将目标函数按照叶子节点展开:
$$
\begin{align}
Obj^t &≈ \sum_{i=1}^n[l(y_i,\hat{y}^{(t-1)}+g_if_t(x_i)+\frac{1}{2}f_t^2(x_i))]+\Omega(f_t) \\
&=\sum_{i=1}^n[l(y_i,\hat{y}^{(t-1)}+g_if_t(x_i)+\frac{1}{2}f_t^2(x_i))]+\gamma T + \frac{1}{2}\lambda\sum_{j=1}^T w_j^2 \\
&=\sum_{j=1}^T[(\sum_{i \in I_j} g_i)w_j + \frac{1}{2}(\sum_{i \ in I_j}h_i + \lambda)w_j^2]+\gamma T
\end{align}
$$
$$
w_j^* = -\frac{G_j}{H_j+\lambda} \\ Obj = -\frac{1}{2}\sum_{j=1}^T \frac{G_j^2}{H_j+\lambda}+ \gamma T
$$