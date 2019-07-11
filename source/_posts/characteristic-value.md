---
title: 不同特征值的特征向量线性无关
date: 2019-01-10 14:18:16
category: 数学
tags: [矩阵]
description: 证明n阶方针不同特征值的特征向量线性无关
---

证明n阶方针不同特征值的特征向量线性无关
<!--more-->

#命题
假设$n$阶方阵$A$,有$s$个不同特征值$\lambda_1,\lambda_2,\lambda_3...\lambda_s$,对应于$s$个特征向量$\alpha_1,\alpha_2,\alpha_3...\alpha_s$,试证明这$s$个向量线性无关.
#证明
假设$\lambda_1<\lambda_2<\lambda_3...\lambda_s$,若线行无关则必然存在:$$k_1\alpha_1+k_2\alpha_2+k_3\alpha_3+...+k_s\alpha_s=0 \tag{1}$$
其中$k_1,k_2,k_3...k_s$不全为0.
左乘$A$得:
$$k_1A\alpha_1+k_2A\alpha_2+k_3A\alpha_3+...+k_sA\alpha_s=0  \tag{2}$$
即:
$$k_1\lambda_1\alpha_1+k_2\lambda_2\alpha_2+k_3\lambda_3\alpha_3hexo+...+k_s\lambda_s\alpha_s=0 \tag{3}$$
由(1)得:
$$k_1\alpha_1=-(k_2\alpha_2+k_3\alpha_3+...+k_s\alpha_s)\tag{4}$$
带入(3)中得:
$$
k_2\alpha_2(\lambda_2-\lambda_1)+k_3\alpha_3(\lambda_3-\lambda_1)+...+k_s\alpha_s(\lambda_s-\lambda_1)=0
 \tag{5}$$
由于:$\lambda_1<\lambda_2<\lambda_3...\lambda_s$
且其中$k_1,k_2,k_3...k_s$不全为0.$\alpha_1,\alpha_2,\alpha_3...\alpha_s$为非0向量,故(5)式不可能成立,由反证法得(1)式也不可能成立.