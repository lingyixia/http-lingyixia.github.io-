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
>>batch normalization解决的是梯度消失问题,残差解决的是网络增加深度带来的退化问题
目的是学习F(x)=x，但是神经网络学习这个不容易，但是学习F(x)=0更容易(一般初始化的时候都是以0为均值).
在假设现在的目的是把5学到5.1，不加残差网络的参数变化是F(5)->5.1,加残差网络的参数变化是H(5)=F'(5)+5=5.1,也就是F'(5)=0.1，也就是学习的过程变成了学习残差,而残差一般数值较小,神经网络对小数更敏感。
平常的前向公式:
$$
x_L = \sum_{i=1}^LF(x_i,w)
$$
反向传播公式:
$$
\frac{\partial L}{\partial x_l}=\frac{\partial L}{\partial O} \prod_l \frac{\partial O}{\partial x_l} 
$$
残差网络的前向公式:
$$
x_L = x_l+\sum_{i=1}^LF(x_i,w)
$$
反向传播公式:
$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial O}(1+ \prod_l \frac{\partial O}{\partial x_l})
$$
[残差网络名字由来](https://zhuanlan.zhihu.com/p/42706477)

##[CNN复杂度分析](https://zhuanlan.zhihu.com/p/31575074)

# RNN
##[RNN梯度消失和爆炸](https://www.zhihu.com/question/34878706)
>>回答要点:1,所说的梯度消失指的是远距离的消失，近处的不会消失
2.所谓的解决是在C的更新路径上解决，而h的路径上，该消失还会消失。

##LSTM公式、结构，s和h的作用
LSTM和普通的RNN相比,多了三个门结构，用来解决当序列过长造成的梯度爆炸或梯度消失问题(写下三个门结构的公式),这三个门结构都是针对输入信息进行处理的,首先对于输入信息要做一个非线性化(非线性化公式),也就是说f是针对上一步的信息拿过来多少，所以我觉得叫记住门更合适，i是真对当前信息留下多少,最后一个ht，也就是说，h是用来保保持前后联系的状态，c是用来维持信息的状态.
##Transform
##参数计算
##CNN和RNN反向传播的不同
[CNN反向传播](https://www.cnblogs.com/pinard/p/6519110.html)需要解决的问题是Pool压缩数据、卷积的解决,Pool可以进行上采样，得到原先的size，卷积可以转为矩阵乘矩阵的形式,比如inputsize为x=9\*9,卷积核为4\*4,stride为1,则输出为5\*5,可以将该卷积操作写为25 \* 81 乘以81 \* 1,然后将得到的25写作5 \*5即可,它每一层更新的都是不同的参数.而RNN由于每个步骤都是共享的参数,因此需要每一个步骤都反向传播到最开始，然后把每个步骤传播过来的梯度相加在更新.

#[overfitting](https://lingyixia.github.io/2019/03/10/neuralNetWorkTips/)
#[BN的作用](https://lingyixia.github.io/2019/03/10/neuralNetWorkTips/)
#[优化方式](https://lingyixia.github.io/2019/03/10/neuralNetWorkTips/)
##激活函数(注意bert的gleu)

#指标介绍
本来很简单的东西解释的一踏糊涂，，还是准备一下措辞吧:
语言介绍:精确度率指的是预测为正例中预测正确的比重 准确率指的是所有样本中预测正确的比重,前者针对正例，后者针对预测正确(包括正例预测正确和负例预测正确)

ROC曲线:
横坐标:$FPR=\frac{FP}{所有负例}$
纵坐标:$TPR=\frac{TP}{所有正例}$(zh)召回率
#[机器学习](https://www.zhihu.com/question/59683332/answer/281642849)
[最大熵模型介绍](https://lingyixia.github.io/2019/07/28/maxlikehood/):第一,充分考虑,第二,不做任何假设
SVM和LR：
相同点:1.都是线性分类器(SVM不考虑核函数)(其实线性的意思是各个特征的线性组合而不是幂次方组合)
      2.都是判别模型
      3.都是有监督模型
不同点:1.损失函数(合页损失核交叉熵损失)
      2.LR优化考虑所有的点，SVM其实最终考虑的只有支持向量
      3.SVM基于距离分类，而LR基于概率分类，所以SVM最好先对数据进行归一化处理，而LR不受影响
      4.SVM的损失函数自带正则，而LR必须另外在损失函数外添加正则项。
      5.SVM解决非线性问题用核函数，LR不用核函数(计算所有数据复杂度太高)会在数据处理上下功夫，比如[数据离散化](https://blog.csdn.net/u011086367/article/details/52879531).
使用场景:
1.异常点多，优先使用逻辑回归(使用所有的数据计算loss，会减少异常点的贡献)
2.特征维度很大，优先使用逻辑回归，因为特征纬度大用参数模型的逻辑回归表达能力更强,而且速度快
3.如果特征维度小，优先使用svm+核函数,因为特征少容易造成非线性问题
4.特征小，数据很多很多，先想办法增加维度，然后用lr或线性svm(svm不擅长处理特征维度大核数据多的问题，花费时间会猛增))

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

#[为什么沿梯度方向最快](https://lingyixia.github.io/2019/06/14/derivative/)

>>简单的说就是快不快要用方向导数定义，而方向导数沿梯度方向最大(两向量共线乘积最大)

#bert点
1. transform的encoder
2. 预测15%,80%mask,10% 替换 10%保留
>>其实每次选取15%来predict对于一个句子来说还是比较多的，很容易造成某个词一直都是mask状态，也就是说最后的模型没有该词的信息,
3. [激活函数](https://blog.csdn.net/liruihongbob/article/details/86510622)(在relu中加入了随机正则)

#L1和L2
L1正则项更容易得到稀疏矩阵，用于特征选择
参考1(https://blog.csdn.net/jinping_shi/article/details/52433975)
[参考2](https://blog.csdn.net/autocyz/article/details/76511527)
L2用于过拟合
[参考3](https://blog.csdn.net/haidixipan/article/details/83186850)