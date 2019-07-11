---
title: 迁移学习综述论文笔记
date: 2019-04-23 10:32:22
category: 深度学习
tags: [迁移学习]
---
>>[迁移学习资料](https://github.com/jindongwang/transferlearning)。迁移学习与普通机器学习、深度学习最大的不同在于放宽了训练数据和测试数据必须满足**独立同分布**的假设和解决有标签训练样本匮乏的问题.

1. [迁移学习研究进展](https://pan.baidu.com/s/1bpautob)
* 根据按源领域和目标领域样本是否标注以及任务是否相同划分分类:
    * 目标领域中有少量标注样本的**归纳迁移学习**
    * 只有源领域中有标签样本的**直推式迁移学习**
    * 源领域和目标领域都没有标签样本的**无监督迁移学习**
* 按采用的技术分类:
    * 半监督学习(半监督学习、直推式学习和主动学习)
    * 基于特征选择方法(选择source domain和 target domain共有特征)
    * 基于特征映射方法(将source domain和target domain映射到高纬度)
    * 基于权重方法(赋予source domain和target domain样本不同的权重)
2. [A Survey on Transfer Learning](https://pan.baidu.com/s/1gfgXLXT)
* 名词解释:domain和task
* 三个问题(重点是1和3):
    1. 迁移什么:提出了迁移哪部分知识的问题; 
    2. 何时迁移:提出了哪种情况下迁移手段应当被运用
    3. 如何迁移:迁移学习的方式
* 迁移什么
![](\img\transferwhat.png)
* 如何迁移:
![](\img\transfer1.png)
* 两者关系
![](\img\transfer2.png)