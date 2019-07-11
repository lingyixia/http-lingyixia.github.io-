---
layout: post
title: 两种交叉熵损失函数对比
date: 2019-01-08 22:00:12
tags: [熵]
description: 两种交叉熵损失函数的对比
---
两种交叉熵损失函数对比
<!--more-->
#第一种
$$
C=\sum_{i=0}^N p(x)\log q(x)
$$
该形式交叉熵损失函数对应神经网络的输出为$softmax$,即这N个$p(x)$加和为1.
#第二种
$$
C=\sum_{i=0}^N p(x)\log q(x)+(1-p(x))\log(1-q(x))
$$
该形式交叉熵损失函数对应神经网络的输出为$sigmod$,即这N个$p(x)$加和不是1.