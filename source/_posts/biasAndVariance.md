---
title: 方差和偏差
date: 2019-03-18 14:36:26
category: 机器学习
tags: [优化]
---

>学习算法的预测误差, 或者说泛化误差(generalization error)可以分解为三个部分: 偏差(bias), 方差(variance)和噪声(noise). 在估计学习算法性能的过程中,我们主要关注偏差与方差. 因为噪声属于不可约减的误差 (irreducible error).

#符号说明
|符号|含义|
| ------ | ------ |
|x|测试样本|
|D|数据集|
|$y_D$|x在D中的标记|
|y|x的真实标记|
|f|训练集D上得到的模型|
|f(x;D)|训练集D上得到的模型在x下输出|
|$\bar{f(x)}$|模型f对x的期望|
>>1.y和$y_D$是有区别的,有些本来就标错了,即噪声。2.这是在不同的训练集D上得到的不同模型,下面的计算也是在不同训练集D下的均值。

#误差定义
* 泛化误差:
$$
Err(x)=E[(y-f(x;D))^2] \tag{1}
$$
* 均值
$$
\bar{f(x)}=E_D[f(x;D)] \tag{2}
$$
>>注意:方差和真实标记y无关,只和模型计算的实际值有关。

* 方差
$$
var(x)=E_D[(f(x;D)-\bar{f(x)})^2] \tag{3}
$$

* 噪声
$$
\epsilon ^2=E_D[(y_D-y)] \tag{4}
$$

* 偏差
$$
bias ^2(x) = (\bar{f(x)}-y)^2 \tag{5}
$$

对算法的期望泛化误差进行分解:
![](/img/bias-variance-proof.png)
其中红色部分计算会得0:
1.
$$
E_D[2(f(x;D)-\bar{f(x)})(\bar{f(x)}-y_D)] = E_D(f(x;D)\bar{f(x)}-\bar{f(x)^2}+\bar{f(x)}y_D-f(x;D)y_D)
$$
要谨记:其中带D得是变量,不带D的都是常量,也就是说$\bar{f(x)}$其实就是个数字,故有:
$$
E_D(f(x;D)\bar{f(x)})=\bar{f(x)}E_D(f(x;D))=\bar{f(x)}^2
$$
前两项得零,后两项等于:
$$
\bar{f(x)}E_D(y_D)-E_D(f(x;D))E_D(y_D)
$$
也等于零
>>$E(XY)=E(X)E(Y)$的必要条件是X和Y相互独立,$E_D(f(x;D))E_D(y_D)$中$f(x;D)$和E_D(y_D)相互独立,因此可以分开。

2.
第二部分怎么得零的不太明白,难道是展开后$E_D(y)=E_D(y_D)$?
整理如图:
![](/img/bias-variance.png)

#方差与偏差
以线性回归为例(训练集(train)、验证集(cv)、测试集(test)比例为6:2:2)
我们现在做的是在训练10个模型,次数依次从1到10,对于 多项式回归,当次数选取较低时,我们的 训练集误差 和 交叉验证集误差 都会很大;当次数选择刚好时,训练集误差 和 交叉验证集误差 都很小;当次数过大时会产生过拟合，虽然 训练集误差 很小,但 交叉验证集误差 会很大(关系图如下)。 图像如下:
![](/img/Degree.jpg)
>>Degree是次数

#正则化参数$\lambda$
对于 正则化 参数,使用同样的分析方法,当参数比较小时容易产生过拟合现象,也就是高方差问题。而参数比较大时容易产生欠拟合现象,也就是高偏差问题.
![](/img/reularization.jpg)
#学习曲线
学习曲线 的横轴是样本数，纵轴为 训练集 和 交叉验证集 的 误差。所以在一开始，由于样本数很少，Jtrain(θ)Jtrain(θ) 几乎没有,而 Jcv(θ)Jcv(θ) 则非常大。随着样本数的增加,Jtrain(θ)Jtrain(θ) 不断增大,而 Jcv(θ)Jcv(θ) 因为训练数据增加而拟合得更好因此下降。所以 学习曲线 看上去如下图:
![](/img/curve1.jpg)
在高偏差的情形下,Jtrain(θ) 与 Jcv(θ) 已经十分接近，但是 误差 很大。这时候一味地增加样本数并不能给算法的性能带来提升。 
![](/img/curve2.jpg)
在高方差的情形下,Jtrain(θ) 的 误差 较小,Jcv(θ) 比较大,这时搜集更多的样本很可能带来帮助。
![](/img/curve3.jpg)
#总结
1. 获得更多的训练样本——解决高方差
2. 尝试减少特征的数量——解决高方差
3. 尝试获得更多的特征——解决高偏差
4. 尝试增加多项式特征——解决高偏差
5. 尝试减少正则化程度λ——解决高偏差
6. 尝试增加正则化程度λ——解决高方差