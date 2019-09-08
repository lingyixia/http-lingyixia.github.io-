---
title: SVM
date: 2019-04-08 14:25:07
category: 机器学习
tags: 
---
svm总结

# 拉格朗日对偶性
$$
\min_x \quad f(x)\\
s.t.\quad  c_i(x)<=0,i=1,2,3...k \\
h_j(x)=0,j=1,2...,l
$$
首先引进拉格朗日函数:
$$
L(x,\alpha,\beta)=f(x)+\sum_{i=1}^k\alpha_ic_i(x)+\sum_{j=1}^l\beta h_j(x)
$$
>>其中,$\alpha_i$>=0

#原始问题
$$
\theta_P(x)=\max_{\alpha,\beta:\alpha_i>=0}L(x,\alpha,\beta)
$$
定义拉格朗日极小极大问题为:
$$
\min_x \theta_P(x)=\min_x\max_{\alpha,\beta:\alpha_i>=0}L(x,\alpha,\beta)
$$
定义原始问题的最优值:
$$
p^*=\min_x \theta_P(x)
$$

##对偶问题
$$
\theta_D(\alpha,\beta)=\min_xL(x,\alpha,\beta)
$$
定义拉格朗日极大极小问题为:
$$
\max_{\alpha,\beta:\alpha_i>=0}\theta_D(\alpha,\beta)=\max_{\alpha,\beta:\alpha_i>=0}\min_xL(x,\alpha,\beta)
$$
定义对偶问题的最优值:
$$
d^*=\max_{\alpha,\beta:\alpha_i>=0}\theta_D(\alpha,\beta)
$$

##原始问题和对偶问题的关系
很显然:
$$
d^*=\max_{\alpha,\beta:\alpha_i>=0}\min_xL(x,\alpha,\beta)<=\min_x\max_{\alpha,\beta:\alpha_i>=0}L(x,\alpha,\beta)=p^*
$$
##KTT条件
$$
\Deta
$$

#硬间隔最大化
假设最终分离超平面表达式为:
$$
wx+b=0
$$
则任意一点到该超平面的**函数间隔**为:
$$
\hat{\gamma_i}=y_i(wx_i+b)
$$
SVM的基本思维就是使任意一点到超平面的最小距离最大,但是**函数距离**不行,需要使用**几何距离**:
$$
\gamma_i=y_i(\frac{w}{||w||}x_i+\frac{b}{||w||}) \\
\gamma=\min_{i=1,2,3...N}\gamma_i
$$
现在可以得到优化表达式:
$$
\max_{w,b} \quad  \gamma \\
s.t. \quad  y_i(\frac{w}{||w||}x_i+\frac{b}{||w||})>=\gamma \quad i=1,2...N
$$
试想,其实$\gamma$无论是多少对最终值都没有任何影响,毕竟$w$和$b$是可以等比例变化的,现在我们假设$\gamma$=$\frac{1}{||w||}$得到:
$$
\max_{w,b} \quad \frac{1}{||w||} \\
s.t. \quad y_i(wx_i+b)>=1
$$
为了向拉格朗日函数靠拢,改写为:
$$
\min_{w,b} \quad \frac{1}{2}||w||^2 \\
s.t. \quad y_i(wx_i+b)-1>=0 \quad i=1,2,3...N
$$
拉格朗日函数:
$$
L(w,b,\alpha)=\frac{1}{2}||w||^2+\sum_{i=1}^N \alpha_i-\sum_{i=1}^N\alpha_iy_i(wx_i+b)
$$
SVM原始问题:
$$
\min_x \theta_P(w,b)=\min_{w,b}\max_{\alpha}L(w,b,\alpha)
$$
对偶问题:
$$
\max_{\alpha}\min_{w,b}L(w,b,\alpha)
$$
现在要求对偶问题的解:
先求$\min_{w,b}L(w,b,\alpha)$,由拉格朗日多元函数条件极值可知,极值必然在各个变量导数为0的位置:
$$
\nabla_wL(w,b,\alpha)=0 \\
\nabla_bL(w,b,\alpha) = 0\\
$$
得到:
$$
w=\sum_{i=1}^N\alpha_iy_ix_i \\
\sum_{i=1}^N\alpha_iy_i=0
$$
从$w$的公式可知模型参数$w$可以完全用训练数据和$\alpha$计算得出。模型在优化的过程中保存的是参数$\alpha$,优化完$\alpha$后可以直接算出$w$.
将$w$的公式带回$L(w,b,\alpha)$,则现在:
$$
\min_{w,b}L(w,b,\alpha)=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_ix_j)+\sum_{i=1}^N\alpha_i
$$
现在自然要求的是极大化上诉公式,即$\max_{\alpha}\min_{w,b}L(w,b,\alpha)$:
$$
\min_{\alpha} \quad \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_ix_j)-\sum_{i=1}^N\alpha_i \\
s.t. \quad \sum_{i=1}^N\alpha_iy_i=0 \\
\alpha_i>=0 \quad i=1,2,3...N
$$
剩下的就是SMO算法了,不在这里详解.
#软间隔最大化
对每个点到分离超平面的距离不在硬性要求大于1,而是引入一个松弛变量。
$$
\min_{w,b} \quad \frac{1}{2}||w||^2 +C\sum_{i=1}^N \xi_i \\
s.t. \quad y_i(wx_i+b)-1 + \xi_i>=0 \quad i=1,2,3...N \\
\xi_i>=0 \quad i=1,2,3...N
$$
计算过程不在详细说明,跟上面差不多,就是多了一个优化参数。
参数说明:$C$越大表示对误分类的惩罚越大，即允许分错的样本越少
#合页损失函数
$$
\sum_{i=1}^N[1-y_i(wx_i+b)]_++\lambda||w||^2
$$
下面证明合页损失函数和软间隔最大化优化公式等价.
令:
$$
[1-y_i(wx_i+b)]_+=\xi_i
$$
必然有$\xi_i$>=0,可以看出,当$1-y_i(wx_i+b)>0$时,$y_i(wx_i+b)=1-\xi_i$,当$1-y_i(wx_i+b)<=0$时,$\xi_i=0$,故$y_i(wx_i+b)>=1-\xi_i$恒成立。故合页损失函数实际上可以写作:
$$
\min_{w,b} \sum_{i=1}^N\xi_i+\lambda||w||^2
$$
取$\lambda=\frac{1}{2C}$得:
$$
\min_{w,b}\frac{1}{C}(\frac{1}{2}||w||^2+C\sum_{i=1}^N\xi_i) \\
s.t. \quad y_i(wx_i+b)>=1-\xi_i \\
\xi_i>=0 \quad i=1,2,3...N
$$
故合页损失函数和软间隔最大化其实等价.
#核函数
核函数是解决非线性支持向量机的方式,在得软间隔的对偶问题:
$$
\min_{\alpha} \quad \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_ix_j)-\sum_{i=1}^N\alpha_i \\
s.t. \quad \sum_{i=1}^N\alpha_iy_i=0 \\
0=<\alpha_i<=C \quad i=1,2,3...N
$$
之后更改为:
$$
\min_{\alpha} \quad \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_ix_j)-\sum_{i=1}^N\alpha_i \\
s.t. \quad \sum_{i=1}^N\alpha_iy_i=0 \\
\alpha_i>=0 \quad i=1,2,3...N
$$
其中$K(x_i,x_j)$就是核函数,用来处理不可分问题,将数据映射到高维.这样最终的分离超平面为:
$$
\sum_{i=0}^N\alpha_iy_iK(x,x_i)+b=0
$$