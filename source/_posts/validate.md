---
title: 简单模型选择
date: 2019-02-27 21:27:13
category: 机器学习
tags:
---

1. 验证集
数据量足够大的前提下可以划分为训练集、验证集和测试集，验证集用于对使用训练集训练得到的模型的验证，然后根据这个验证结构调整模型.
2. 交叉验证
   * 简单交叉验证:
   ```
   while(N)//重复N次,得到N个模型
   {
       while(M)//每个模型验证M次
       {
           将数据打乱
           随机取一部分,比如70%用于训练集,30%用于测试集
           用上诉训练集在参数为i的条件下训练模型
           M--
       }
       计算M次的平均得分,也就是参数为i情况下该模型的得分
       i++ //i是个参数集,不同的i代表不同的参数集
       N--
   }
   取得分最高的模型
   ```
   * S折交叉验证
   ```
   将所有数据均为为S分
   while(N)
   {
       while(S)
       {
           取S-1分用于训练集,剩下的1份用于测试集
           用上诉训练集在参数为i的条件下训练模型
       }
       计算M次的平均得分,也就是参数为i情况下该模型的得分
       i++ //i是个参数集,不同的i代表不同的参数集
       N--
   }
   取得分最高的模型
   ```
   * 留一交叉验证
   ```
   是S交叉验证中S=N,即只用一个样本作为测试集,其他的都是训练集
   ```
   简单交叉验证用于初步分析,S折交叉验证用于深入分析,留一交叉验证用于数据量少的时候.