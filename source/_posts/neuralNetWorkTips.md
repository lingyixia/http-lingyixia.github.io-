---
title: 神经网调优
date: 2019-03-10 10:13:05
catagory: 深度学习
tags: [tips]
---
# OverFitting
1. early stop
2. 正则化
3. Dropout
4. 增加数据量

tip:为什么正则化防止过拟合?
$$
L=L_0+\frac{\lambda}{2n}\sum w^2 \\
\frac{\partial L}{\partial w}=\frac{\partial L_0}{\partial w}+ \frac{\lambda}{n} w \\
\frac{\partial L}{\partial b}=\frac{\partial L_0}{\partial b}
$$
可以看出,正则项对b没影响,对w有影响,即:
$$
w=(1-\frac{\eta \lambda}{n}) w -\eta \frac{\partial L_0}{\partial w}
$$
也即是说w**权重衰减**了,w更小了,为什么w更小了就能防止过拟合呢?因为过拟合的网络一般参数较大,w更小了网络就更简单了,所谓奥卡姆剃刀,所以过拟合减少了.

# 数据预处理
中心化\标准化

#权重初始化
1.随机初始化为小数据,如均值为0,方差为0.01的高斯分布,初始化小数据是为了打破对称性
2.(1/sqrt(n))未完待续

# 加速训练

1. SGD (训练集做手脚)
2. momentum(在梯度上做手脚)
$$
v_t=\lambda v_{t-1}+(1-\lambda)_tg_t \\
w_t=w_{t-1}-v_t
$$
>>初始化$v_0$=0当前梯度是前面所有梯度的加权平均,当前时刻计算的梯度的作用仅仅是在前面梯度加权平均上的调整

3. Adagard (学习率做手脚)
本来应该:
$$
w^{t+1} \leftarrow w^t - \eta^t g^t
$$
Adagard:
$$
s_t =s_{t-1}+g_t^2 \\
w_t=w_{t-1} - \frac{\eta}{\sqrt{s_t+\epsilon}} \times g_t
$$
>>Adagard中对每个参数都有不同的学习率,比如一个参数$W_1$的梯度较大,则经过这个计算之后学习率就较小,反之$w_2$的梯度较小,计算之后学习率较大。

4. RMSProp(3的改进)
$$
s_t=\lambda s_{t-1}+(1-\lambda)g_t^2 \\
w_t=w_{t-1} - \frac{\eta}{\sqrt{s_t+\epsilon}} \times g_t
$$
>>初始化$s_0$=0
5. Adadelta(没有学习率)
$$
s_t= \rho s_t+(1-\rho)g_t^2 \\
g_t^{\prime}=\sqrt{\frac{\Delta w_{t-1}+\epsilon}{s_t+\epsilon}}*g_t \\
w_t= w_{t-1}-g_t^{\prime} \\
\Delta w_t=\rho \Delta w_{t-1}+(1-\rho)g_t^{\prime} * g_t
$$
>>其实就是维护了一个$\Delta w$用来代替Adagard中的学习率。

6. Adam(2和4的结合)

$$
v_t=\beta_1 v_{t-1} +(1-\beta_1)g_t \\
s_t=\beta_2 s_{t-1} + (1-\beta_2)g_t^2\\
\hat{v_t}=\frac{v_t}{1-\beta_1^t} \\
\hat{s_t}=\frac{s_t}{1-\beta_2^t} \\
g_t^{\prime} = \frac{\eta \hat{v_t}}{\sqrt{\hat{s_t}}+\epsilon} \\
w_t=w_{t-1}-g_t^{\prime}
$$
>>初始化$v_0$=0,$s_0$=0其中3、4两步是参数修正,原因是如果不这样刚开始的时候有可能会使v和g,S和g相差太大.