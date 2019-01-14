---
title: word2vec
date: 2019-01-10 18:00:04
category: 深度学习
tags: [词向量]
mathjax: true
description: 词向量小记
---
>本文不作为详细教程，知识自己容易遗忘知识点的记录。

#语言模型
通俗来说就是用来量化哪个句子更像人话的模型,目前可分为统计语言模型和神经网络模型。
#统计语言模型
由Bayes公式,一个句子组成的概率公式为:
$$
p(w_k|w_1^{k-1})=\frac{p(w_1^k)}{p(w_1^{k-1})} \tag{1}
$$
根据大数定理:
$$
p(w_k|w_1^{k-1})≈\frac{count(w_1^k)}{count(w_1^{k-1})} \tag{2}
$$
由**马尔科夫假设:任意一个词出现的概率只是他前面出现的有限的一个或者几个词相关(未来的事件，只取决于有限的历史)**
基于马尔可夫假设，N-gram 语言模型认为一个词出现的概率只与它前面的n-1个词相关,即:
$$
p(w_k|w_{k-n+1}^{k-1})=\frac{p(w_{k-n+1}^k)}{w_{k-n+1}^{k-1}}≈\frac{count(w_{k-n+1}^k)}{count(w_{k-n+1}^{k-1})} \tag{3}
$$
eg:
n=2
$$
p(w_k|w_{k-1})≈\frac{count(w_{k-1}^{k})}{count(w_{k-1})} \tag{4}
$$
一般而言,语言模型利用最大似然确定目标函数(即最大化某句子的概率):
$$
Loss=\sum_w \log p(w|context(w))  \tag{5}
$$
其中:
$$
p(w|context(w))=F(w,context(w),\theta)  \tag{6}
$$
比如早期的智能ABC据说就是用的N-gram,还有搜索引擎,输入一个词后面多个选项也是N-gram.
#神经网络语言模型
##神经概率语言模型(NPLM)
该结构是词向量模型的先驱
网络结构:
![](/img/NPLM.png)
* Input Layer是n-1个词的词向量**首尾相连**,得到$x_w$,长度为:$(n-1)m$,$m$是词向量长度.
* Projection Layer是:$z_w=tanh(Wx_w+p)$,长度依然为$(n-1)m$.
* Hidden Layer是$y_w=Uz_w+q$,得到的$y_w=(y_1,y_2...y_N)$,N是词表大小.
* Output Layer是$softmax$,即将$y_w$的各个分量做$softmax$，使其加和为1.

由式(6)$F(w,context(w),\theta)$,此时$\theta$包含:
* 词向量: $v_w\in R^m$
* 神经网络参数: $W \in R^{n_h \times (n-1)m},p \in R^{n_h},Y \in R^{n_h},q \in R^{N}$
其中$n_h$是batchsize,N是词典大小。


##词向量模型
###基于Hierarchical SoftMax
####CBOW
由上下文词推断当前词的词向量模型
![](/img/hcbow.png)
* Input Layer: $2c$个词的随机初始化词向量
* Projection Layer是:$x\_w=\sum_{i=1}^{2c}v(context(w)_i)$
* Output Layer是一颗由训练语料构成的Huffman树

与NPML相比,他们的$x_w$不同,NPML是收尾相接,而$CBOW$是向量加和.他们的输出层不同,NPML是输出层是线性结构,$CBOW$输出层是$Huffman$树.
损失函数计算:
引入一些符号:
* $p^w$:从根节点到达$w$节点的路径
* $l^w$:路径$p^w$中节点的个数
* $p^w_1$...$p^w_{lw}$
####Skip-gram