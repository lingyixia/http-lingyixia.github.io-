---
title: 卷积核的选择
date: 2019-02-09 20:32:40
tags: [CNN, 卷积核]
category: 深度学习
---
>CNN中卷积核的选择问题

#感受视野
![](/img/kernel1.png)
>>图中三个卷积核都是$3 \times 3$,第一个卷积核的感受视野是$3 \times 3$

![](/img/kernel2.png)
>>第二个卷积核的感受视野是$5 \times 5$

![](/img/kernel3.png)
>>第三个卷积核的感受视野是$7 \times 7$

#大小卷积核
以图二为例,用两层$3 \times 3$的卷积核所需的参数数量是$(3 \times 3 +1) \times 2 \times n$,使用一层$5 \times 5$,参数数量为$(5 \times 5 +1 ) \times n$,两者感受野相同,但是明显前者参数数量少.