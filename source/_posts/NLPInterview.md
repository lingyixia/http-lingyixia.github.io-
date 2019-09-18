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
##CNN感受野为什么都是奇数:
1.从SameAndValid](https://lingyixia.github.io/2019/03/13/CnnSizeCalc/),Padding = Kernel_size-1,当Kernel_size为奇数的时候可以平均到两边
2.方便确定卷积核的位置，中心点就可以代表卷积核的位置，但是偶数的话就不方便
##Pooling的作用
##[卷积和池化的区别](https://mp.weixin.qq.com/s/MCIhRmbq2N0HM_0v-nld_w)
##参数计算方法
##输出层维度计算
##[1*1卷积的作用](https://mp.weixin.qq.com/s/MCIhRmbq2N0HM_0v-nld_w)
##反卷积和空洞卷积
##(残差网络)[https://lingyixia.github.io/2019/04/05/transformer/]
##transformer
两个问题:
1,为什么用多头:多个attention便于模型学习不同子空间位置的特征表示，然后最终组合起来这些特征，而单头attention直接把这些特征平均，就减少了一些特征的表示可能。 
2.为什么self-attention要除以一个根号维度:假设Q和K都是独立的随机变量，满足均值为0，方差为1，则点乘后结果均值为0，方差为dk。也即方差会随维度dk的增大而增大，而大的方差导致极小的梯度(我认为大方差导致有的输出单元a（a是softmax的一个输出）很小，softmax反向传播梯度就很小（梯度和a有关））。为了避免这种大方差带来的训练问题，论文中用内积除以维度的开方，使之变为均值为0，方差为1。

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

##LSTM和GRU
GRU和LSTM的性能在很多任务上不分伯仲。
GRU 参数更少因此更容易收敛，但是数据集很大的情况下，LSTM表达性能更好。
从结构上来说，GRU只有两个门（update和reset），LSTM有三个门（forget，input，output），GRU直接将hidden state 传给下一个单元，而LSTM则用memory cell 把hidden state 包装起来。
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
[绘制过程](https://blog.csdn.net/program_developer/article/details/79946787)
横坐标:$FPR=\frac{FP}{样本标签所有负例}$，也就是所有的负例有多少被预测为正例了.
纵坐标:$TPR=\frac{TP}{样本标签所有正例}$(zh)召回率,也就是所有的正例有多少被预测为正例了.
 AUC是衡量二分类模型优劣的一种评价指标，表示预测的正例排在负例前面的概率。

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
3.如果特征维度小，数据少,优先使用svm+核函数,因为特征少容易造成非线性问题,而数据少只要支持向量没变最后的平面也不会变。
4.特征小，数据很多很多，先想办法增加维度，然后用lr或线性svm(svm不擅长处理特征维度大核数据多的问题，花费时间会猛增))

逻辑回归假设服从伯努利分布，线性回归假设服从正太分布
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

#调参不理想
如果超参数调不好就需要
1.考虑参数初始化问题和激活函数
2.看是否overfitting

# Dropout加在哪里
word embedding层后、pooling层后、FC层（全联接层）后

#GBDT和XGboost

#大文件小到大排序
1.将大文件分割为N个内存可读的小文件,每个小文件从小到大排序
2.每个文件取一个指针指向头数据
3.将头数据组成小根堆
4.将堆顶文件存入文件,将之前堆顶文件的下一个数据放入堆顶，调整成小根堆
5.执行步骤4，直到完成.

#类别不均衡
1.调整class-weight，使对少的类更敏感(把类从小到大数量排序，在倒过来排序，softmax后就是各个权重，即少的大，多的小s)
2.比如大类是小类的N倍，则训练N个模型，每个模型都是小类全部和大类的1/N，最后投票。
3.少类过采样(复制多分，过拟合)，多类欠采样(没有充分利用样本)
 * SMOTE可以合成少量数据用来训练

 #GDBT和RF优点缺点

 # [维比特算法](https://wulc.me/2017/03/02/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95/)
 假设序列长度为,隐含状态数量为m，穷举法的时间复杂度为$O(n^m)$
 维比特算法时间复杂度为$O(m*n^2)$
 >>解释，比如从t=1到t=2，需要计算n*n次才能计算的到t=2的每个概率值，要进行m个步骤，因此是$O(m*n^2)$