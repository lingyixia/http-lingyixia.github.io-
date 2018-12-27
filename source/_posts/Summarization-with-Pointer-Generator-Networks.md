---
layout: post
title: Summarization-with-Pointer-Generator-Networks
date: 2018-12-26 19:30:48
category: 深度学习
tags: [论文,tensorflow]
description: Get To The Point:Summarization with Pointer-Generator Networks论文导读
---
>本论文针对的是摘要生成任务,普遍而言文本生成类的任务都是使用Seq2Seq模型,本论文也不例外,一般而言文本生成有两种模式:抽取式和生成式,顾名思义不在详细描述，本论文分为三个部分:传统Attention,Pointer-Generator Networks和Coverage机制.论文连接: https://arxiv.org/pdf/1704.04368.pdf

#传统Attention
简述:
$$ 
\begin{align}
e_i^t&=V^Ttanh(W_hh_i+W_s+b_{attn}) \tag{1} \\
a^t&=sotfmax(e^t) \tag{2} \\
h_t^*&=\sum_ia_i^th_i \tag{3} \\
\end{align}
$$
$$
\begin{gather}
P_{vocab}=sotfmax(V^{'}(V[o_t;h_t^*]+b)+b^{'}) \tag{4} \\
P(w)=P_{vocab}(w) \tag{5} \\
loss_t = -logP(w_t^*) \tag{6} \\
loss=\frac{1}{T}\sum_t^Tloss_t \tag{7}
\end{gather}
$$
其中(4)中$o_t$是`LSTM`的输出,外面包着的两层实际上是个全连接网络$V^{'},V,b,b^{'}$也就是两个全连接网络的参数。
需要注意的是代码实现方式,见[attention_decoder.py](https://github.com/becxer/pointer-generator/blob/master/attention_decoder.py)
由于`h`的`shape`是`(batch_size,encode_length,hidden_size)`，无法直接加权重`W`,
作者首先增加其维度,得到`(batch_size,encode_length,1,hidden_size)`利用了`conv2d`卷积函数实现,得到相同`shape`,而$s_t$的`shape`是`(batch_size,hidden_size)`,要加到每个`encode_length`上,作者首先增加其维度，得到`shape`为`(batch_size,1,1,hidden_size)`,然后利用**广播机制**,直接相加即可.
#Pointer-generator network
原理很简单，就是在`Attention`步骤中,认为(2)中得到的就是词概率，当然，此时的词概率所计算的词只有`encoder`部分原文的词量，而上面求得的$P_{vocab}$的词量是整个词典的量,但是两者并不是后者包含前者，而是交集关系，因为`encoder`的原文很可能包含原训练集不包含的词语,因此