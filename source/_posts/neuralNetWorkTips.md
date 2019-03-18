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
v_t=\lambda v_{t-1}+\eta_tg_t \\
w_t=w_{t-1}-v_t
$$
>>其思想就是让参数更新方向不那么`硬`

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

4. RMSProp(2和3的结合)
$$
s_t=\lambda s_{t-1}+(1-\lambda)g_t^2 \\
w_t=w_{t-1} - \frac{\eta}{\sqrt{s_t+\epsilon}} \times g_t
$$
5. Adadelta(没有学习率)
$$
s_t= \rho s_t+(1-\rho)g_t^2 \\
g_t^{\prime}=\sqrt{\frac{\Delta w_{t-1}+\epsilon}{s_t+\epsilon}}*g_t \\
w_t= w_{t-1}-g_t^{\prime} \\
\Delta w_t=\rho \Delta w_{t-1}+(1-\rho)g_t^{\prime} * g_t
$$
>>其实就是维护了一个$\Delta w$用来代替Adagard中的学习率。

6. Adam

$$
v_t=\beta_1 v_{t-1} +(1-\beta_1)g_t \\
s_t=\beta_2 s_{t-1} + (1-\beta_2)g_t^2\\
\hat{v_t}=\frac{v_t}{1-\beta_1^t} \\
\hat{s_t}=\frac{s_t}{1-\beta_2^t} \\
g_t^{\prime} = \frac{\eta \hat{v_t}}{\sqrt{\hat{s_t}}+\eta} \\
w_t=w_{t-1}-g_t^{\prime}
$$
>>[好累,先干点别的,明天再继续](http://zh.gluon.ai/chapter_optimization/adam.html)

5. dropout(随机丢弃部分神经元)
tensorflow中的实现方式就是对每个神经元维护一个概率值,每次训练都以该概率的概率将其所有权重置为零