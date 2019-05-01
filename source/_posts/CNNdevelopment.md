---
title: CNN发展史
date: 2019-05-01 18:10:28
category: [深度学习]
tags: [CNN]
---
>简单记录一下CNN发展历程

# [LeNet](https://my.oschina.net/u/876354/blog/1632862)
![](\img\CNNdevelopment\LeNet.jpg)
>>诞生:1998 作者:LeCun 别称:CNN鼻祖

结构:
1. Input层:$32 \times 32 \times 1$
2. C1层:
    * 输入:$32 \times 32 \times 1$
    * kernel:6个$5 \times 5$
    * stride:$1 \times 1$
    * 输出:$28 \times 28 \times 6$
3. S2层:
    * 输入:$28 \times 28 \times 6$
    * pooling:$2 \times 2$
    * 输出:$14 \times 14 \times 6$
4. C3层:
    * 输入:$14 \times 14 \times 6$
    * kernel:16个$5 \times 5$
    * stride:$1 \times 1$
    * 输出:$10 \times 10 \times 16$
5. S4层:
    * 输入:$10 \times 10 \times 16$
    * pooling: $2 \times 2$
    * 输出:$5 \times 5 \times 16$
6. C5层:
    * 输入:$5 \times 5 \times 16$
    * kernel:120个$5 \times 5$
    * stride:$1 \times 1$
    * 输出:$1 \times 1 \times 120$
7. 全连接层F6:
    * 输入:$1 \times 120$
    * 输出:$120 \times 84$(10个种类)
8. 全连接输出层:
    * 输入:$1 \times 84$
    * 输出:$1 \times 10$

# [AlexNet](https://my.oschina.net/u/876354/blog/1633143)
![](\img\CNNdevelopment\AlexNet.jpg)
>>诞生:2012年ImageNet竞赛 作者:Hinton和他的学生Alex Krizhevsky 别称:CNN王者归来

特点:
* ReLU:收敛快,计算量少
* 数据扩充:对256×256的图片进行随机裁剪到224×224，然后进行水平翻转，相当于将样本数量增加了$((256-224)^2)×2=2048$倍,减少overfitting
* 重叠池化:减少overfitting
* dropout:减少overfitting
* LRN
* 多GPU

结构:
1. L1:卷积$\rightarrow$ReLU$\rightarrow$池化$\rightarrow$归一化
    * 输入:$224 \times 224 \times 3$
    * kernel:96个$11 \times 11 \times 3$
    * padding:3
    * stride: 4
    * 卷积输出:$55 \times 55 \times 96$
    * pooling:$3 \times 3$ stride:2
    * 输出:$27 \times 27 \times 96$
2.  L2:卷积$\rightarrow$ReLU$\rightarrow$池化$\rightarrow$归一化
    * 输入:$27 \times 27 \times 96$
    * kernel:256个$5 \times 5 \times 96$
    * padding:两个2
    * stride: 1
    * 卷积输出:$27 \times 27 \times 256$
    * pooling:$3 \times 3$ stride:2
    * 输出:$13 \times 13 \times 256$
3. L3:卷积$\rightarrow$ReLU
    * 输入:$13 \times 13 \times 256$
    * kernel:384个$3 \times 3$
    * padding:两个1
    * stride:1
    * 卷积输出:$13 \times 13 \times 384$
4. L4:卷积$\rightarrow$ReLU
    * 输入:$13 \times 13 \times 384$
    * kernel:256个$3 \times 3$
    * padding:两个1
    * stride:1
    * 卷积输出:$13 \times 13 \times 256$
5. L5:卷积$\rightarrow$ReLU$\rightarrow$池化
    * 输入:$13 \times 13 \times 256$
    * kernel:256个$3 \times 3$
    * padding:两个1
    * stride:1
    * 卷积输出:$13 \times 13 \times 256$
    * pooling:$3 \times 3$ stride:2
    * 输出:$6 \times 6 \times 256$
6. L6:全连接(全卷积)$\rightarrow$ReLU$\rightarrow$Dropout
    * 输入:$6 \times 6 \times 256$
    * 全卷积的意思是4096个$6 \times 6 \times 256$的卷积核
    * 输出:$4096  \times 1 \times 1$
7. L7:全连接$\rightarrow$ReLU$\rightarrow$Dropout
    * 输入输出都是4096
8. L8全连接:
    * 输入4096
    * 输出1000

# [VGGNet](https://my.oschina.net/u/876354/blog/1634322)
![](\img\CNNdevelopment\VggNet.png)
>>诞生:2014年ImageNet竞赛 作者:牛津大学Andrew Zisserman教授的组

特点:
* [小卷积核和多卷积子层](http://lingyixia.github.io/2019/02/09/kernel/)
* 小池化核:相比AlexNet的3x3的池化核，VGG全部采用2x2的池化核。
* 层数更深、特征图更宽:由于卷积核专注于扩大通道数、池化专注于缩小宽和高，使得模型架构上更深更宽的同时，控制了计算量的增加规模。
* 全连接转卷积（测试阶段）
网络结构见链接吧,不写了,但是这里要总结一下$1 \times 1$卷积核的作用:
    * 全卷积其实就是全连接,$1 \times 1$卷积不是全连接
    * $1 \times 1$卷积核前后size不变,通道数可以改变,因此可以用来降维或升维
    * 增加非线性(没明白)
    * 跨通道信息交互