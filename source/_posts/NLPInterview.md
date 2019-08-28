---
title: NLPInterview
date: 2019-05-02 14:20:23
tags:
---
>>总结一些NLP面试必备知识点
链接总结:
1.https://mp.weixin.qq.com/s/MCIhRmbq2N0HM_0v-nld_w

# [CNN](https://lingyixia.github.io/2019/01/23/CNN/)
##[发展史](https://lingyixia.github.io/2019/05/01/CNNdevelopment/)
##[感受野](https://lingyixia.github.io/2019/05/13/DilatedConvolution/)
##[SameAndValid](https://lingyixia.github.io/2019/03/13/CnnSizeCalc/)
##Pooling的作用
##[卷积和池化的区别](https://mp.weixin.qq.com/s/MCIhRmbq2N0HM_0v-nld_w)
##参数计算方法
##输出层维度计算
##[1*1卷积的作用](https://mp.weixin.qq.com/s/MCIhRmbq2N0HM_0v-nld_w)
##反卷积和空洞卷积
##(残差网络)[https://lingyixia.github.io/2019/04/05/transformer/]
##CNN复杂度分析

# RNN
##[RNN梯度消失和爆炸](https://www.zhihu.com/question/34878706)
##LSTM公式、结构，s和h的作用
LSTM和普通的RNN相比,多了三个门结构，用来解决当序列过长造成的梯度爆炸或梯度消失问题(写下三个门结构的公式),这三个门结构都是针对输入信息进行处理的,首先对于输入信息要做一个非线性化(非线性化公式),也就是说f是针对上一步的信息拿过来多少，所以我觉得叫记住门更合适，i是真对当前信息留下多少,最后一个ht，也就是说，h是用来保保持前后联系的状态，c是用来维持信息的状态.
##Transform
##参数计算

#[overfitting](https://lingyixia.github.io/2019/03/10/neuralNetWorkTips/)
#[BN的作用](https://lingyixia.github.io/2019/03/10/neuralNetWorkTips/)
#[优化方式](https://lingyixia.github.io/2019/03/10/neuralNetWorkTips/)
##激活函数(注意bert的gleu)

#指标介绍
本来很简单的东西解释的一踏糊涂，，还是准备一下措辞吧:
语言介绍:精确度率指的是预测为正例中预测正确的比重 准确率指的是所有样本中预测正确的比重,前者针对正例，后者针对预测正确(包括正例预测正确和负例预测正确)
#[机器学习](https://www.zhihu.com/question/59683332/answer/281642849)

#[激活函数](https://www.cnblogs.com/hutao722/p/9732223.html)
>>Relu/sigmoid /tanh
sigmoid: y = 1/(1 + e-x)
tanh: y = (ex - e-x)/(ex + e-x)
relu: y = max(0, x)

激活函数通常有以下性质： 
* 非线性：如果激活函数都为线性，那么神经网络的最终输出都是和输入呈线性关系；显然这不符合事实。 
* 可导性：神经网络的优化都是基于梯度的，求解梯度时需要确保函数可导。 
* 单调性:激活函数是单调的，否则不能保证神经网络抽象的优化问题为凸优化问题了。 
* 输出范围有限：激活函数的输出值的范围是有限时，基于梯度的方法会更加稳定。输入值范围为 (−∞,+∞) ，如果输出范围不加限制，虽然训练会更加高效，但是learning rate将会更小，且给工程实现带来许多困难。

第一,sigmoid和tanh对比,后者相当于前者的平移版本,取值范围是[-1,1],可以看做一种数据中心化的效果
第二,sigmoid最大的优势就是可以输出概率
第三,sigmoid反向求导**计算量大**,**容易梯度消失**,relu速度快，不会梯度消失