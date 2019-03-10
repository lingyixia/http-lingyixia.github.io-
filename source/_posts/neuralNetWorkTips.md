---
title: 神经网调优
date: 2019-03-10 10:13:05
catagory: 深度学习
tags: [tips]
---

# 加速训练

1. SGD (训练集做手脚)

2. momentum(在梯度上做手脚)

假设在t步骤时$W$的**梯度**(为$dW$,$b$的梯度为$db$,t-1步计算得到的**速度**是$V_{dw}$和$V_{db}$,则:
$$
V_{dw}=\beta V_{dw}+(1-\beta)dW \\
V_{db}=\beta V_{db}+(1-\beta)db \\
W=W-\alpha V_{dw} \\
b=b-\alpha V_{db}
$$
>>其思想就是让参数更新方向不那么`硬`

3. Adagard (学习率做手脚)

本来应该:
$$
w^{t+1} \leftarrow w^t - \eta^t g^t
$$
Adagard:
$$
w^{t+1} \leftarrow w^t-\frac{\eta^t}{\sqrt{\sum_{i=0}^t(g^i)^2}} g^t
$$

4. RMSProp(2和3的结合)