---
title: 膨胀卷积
date: 2019-05-13 21:37:06
category: 深度学习
tags: [CNN,卷积核,感受野]
---
>膨胀卷积是为了解决CNN中pooling损失信息而诞生的,CNN中pooling的作用一般有二,其一是为了减少参数,其二是为了扩大感受野

#感受野
>>通俗解释就是当前层的一个单位是由输入层多少单位计算得到的.

eg:
![](\img\kernel1.png)
图右侧的一个单位是输入层9个单位计算得到的,感受野是9

#大小卷积核
>>多层小卷积核能够获得和单层大卷积核相同的感受野,但是大大减少参数量

eg1:
![](\img\kernel2.png)
>>图左侧感受野是$5 \times 5 = 25$,要想达到这个感受野使用一个卷积核只能是$5 \times 5$,即$(5 \times 5 +1) \times n$个参数,但是图中进行了两次卷积,使用了两个卷积核,同样达到了$5 \times 5$的感受野,但是使用了两个$3 \times 3$的卷积核,即参数大小为:$(3 \times 3 +1) \times n + (3 \times 3 +1) \times n$个参数,比原来少了$6n$个参数。

eg2:
![](\img\kernel3.png)
>>图左侧感受野是$7 \times 7 = 49$,要想达到这个感受野使用一个卷积核只能是$7 \times 7$,即$(7 \times 7 +1) \times n$个参数,但是图中进行了三次卷积,使用了三个卷积核,同样达到了$7 \times 7$的感受野,但是使用了三个卷积个$3 \times 3$的卷积核,即参数大小为:$(3 \times 3 +1) \times n + (3 \times 3 +1) \times n + (3 \times 3 +1) \times n$个参数,比原来少了$20n$个参数。

#感受野计算公式
$r = (m-1) \times stride+ksize$,其中,$m$表示上层感受野(初始感受野为1),$r$表示本层感受野,$stride$表示步长,$ksize$表示卷积核大小

#膨胀卷积
普通卷积:
![](\img\normalCNN.gif)
膨胀卷积:
![](\img\dilatedCNN.gif)
##膨胀卷积核计算
![](\img\dilatedKernel.png)
将原始卷积核填充若干行\列0,得到更大的卷积核,卷积的时候stride为1
$n-dilated convolution$,一般$n$是$2$的整数次幂
(a) 普通卷积,$1-dilated convolution$,卷积核的感受野为$3 \times 3$(当n=1的时候也就是普通的CNN)
(b) 扩张卷积,$2-dilated convolution$,卷积核的感受野为$7 \times 7$
(c) 扩张卷积,4-dilated convolution,卷积核的感受野为$15 \times 15$
上诉三个感受野是怎么计算的呢?
首先三个膨胀卷积核大小为:$3 \times 3$,$5 \times 5$,$9 \times 9$
(a) 感受野为:$(1-1) \times 1+3 = 3$
(b) 感受野为:$(3-1) \times 1+5 = 7$
(c) 感受野为:$(7-1) \times 1+9 = 15$

一般情况下第一层先用普通卷积,然后下几层用n-dilated convolution
![](\img\cnnAndDilated.png)
>>可以看出,两者红色部分感受野分别为5和7

[参考一](https://www.cnblogs.com/houjun/p/10275215.html)
[参考二](https://kexue.fm/archives/5409)
[参考三](https://blog.csdn.net/mao_xiao_feng/article/details/78003730)