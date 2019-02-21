---
title: representationMethod
date: 2019-02-21 16:35:12
category: 深度学习
tags: [表征]
mathjax: true
---

>词表征(word representation)和句子表征(sentence representation)方式总结

# 词表征(word representation)
## one-hot encoding
不解释
## Word Embedding
>>主要分为两类:Frequency based Embedding和Prediction based Embedding

### Frequency based Embedding
* Count Vector
最简单的方式,将train用的N个文本按顺序排列,统计出所有文本中的词频,结果组成一个矩阵,那么每一列就是一个向量，表示这个单词在不同的文档中出现的次数。
* TF-IDF Vector
TF-IDF方法基于前者的算法进行了一些改进,它的计算公式如下:
$$
tfidf_{i,j} = tf_{i,j} \times idf_i
$$
其中,$tf_{i,j}$(term-frequence)指的是第i个单词在第j个文档中出现的频次;而$idf_i$(inverse document frequency)的计算公式如下:
$$
idf_i = \log (N/n)
$$
其中,$N$表示文档的总个数,$n$表示包含该单词的文档的数量。这个公式是什么意思呢？其实就是一个权重，设想一下如果一个单词在各个文档里都出现过，那么$N/n=1$,所以$idf_i=0$,这就意味着这个单词并不重要。这个东西其实很简单,就是在term-frequency的基础上加了一个权重，从而显著降低一些不重要/无意义的单词的frequency,比如a,an,the等,很明显$n$越大$idf_i$越小,即所谓的逆文本。

* Co-Occurrence Vector(不重要)
直译过来就是协同出现向量。在解释这个概念之前，我们先定义两个变量:
    * Co-occurrence
协同出现指的是两个单词$w_1$和$w_2$在一个Context Window范围内共同出现的次数
    * Context Window
指的是某个单词$w$的上下文范围的大小,也就是前后多少个单词以内的才算是上下文?

### Word2vec 
上面介绍的三种Word Embedding方法都是确定性(deterministic)的方法,而接下来介绍一种非确定性的基于神经网络的预测模型——word2vec。它是只有一个隐含层的神经网络,且激活函数(active function)是线性的,最后一层output采用softmax来计算概率。它包含两种模型:CBOW和Skip-gram
