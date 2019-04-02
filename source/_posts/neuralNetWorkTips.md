---
title: 神经网调优
date: 2019-03-10 10:13:05
catagory: 深度学习
tags: [tips]
---

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

5. dropout(随机丢弃部分神经元)
tensorflow中的实现方式就是对每个神经元维护一个概率值,每次训练都以该概率的概率将其所有权重置为零