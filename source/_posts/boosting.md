---
title: boosting
date: 2019-02-22 19:04:20
category: 机器学习
tags: [分类,回归]
---
>boosting 是一种特殊的集成学习方法。所有的‘基’分类器都是弱学习器，但通过采用特定的方式迭代，每次根据训练过的学习器的预测效果来更新样本权值,用于新的一轮学习,最终提高联合后的学习效果boosting

#AdaBoost
`AdaBoost`算法中,标签`y`是{-1,1}.
目标函数是:
$$
\begin{align}
f(x) &=sign(F(x)) \\
     &=\sum_{k=1}^K \alpha_kT_k(x;\beta_k)
\end{align}
$$
损失函数定义为:
$$

$$