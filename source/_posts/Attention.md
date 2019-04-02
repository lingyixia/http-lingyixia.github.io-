---
title: Attention小结
date: 2019-04-02 16:20:15
category: 深度学习
tags: [Attention]
---

符号说明:
* 当前是第`t`步
* `s_t`:`decoder`阶段隐状态
* `h_t`:`encoder`阶段隐状态
* `N`:`encoder`步数
#BahdanauAttention
以`encoder-decoder`模型为例,假设当前是第`t`步:
$$
y_t=decoder(y_{t-1},s_{t-1},context_t) \\
context_t = \sum_{i=1}^N a_{ti}·h_i \\
a_{ti}=softmax(\frac{e^{e_{ti}}}{\sum_{j=1}^N e_{e{tj}}}) \\
e_{tj}=scores(s_{t-1},h_j)
$$

#LuongAttention
公式和上面基本一样,只有两点不同:
1. BahdanauAttention在计算scores的时候是用上一个隐藏状态$s_{t-1}$,而LuongAttention使用当前隐藏状态$s_t$
2. 在计算scores的时候BahdanauAttention使用的是`encoder`阶段各层的状态`contact`后计算,而LuongAttention仅计算最上层。

#scores
其实各种Attention不同点都在计算scores的时候作文章下面是几种计算scores的方式:
![](\img\attention.png)

#总结
其实所有的attention都归结为三个点:`Q`,`K`,`V`,所有的attention通用公式其实就是:
$$
Attention(Q,K,V)=f(softmax(scores(Q,K))V)
$$
而且一般`K=V`
在`encoder-decoder`翻译模型中,`Q`指的是译文,`K`和`V`指的是原文.
当`Q=K=V`的时候,其实也就是`self-attention`了,也就是`transform`模型中的关键点:
$$
Attention(Q,K,V)=softmax(\frac{QK^K}{\sqrt{d_k}})V
$$