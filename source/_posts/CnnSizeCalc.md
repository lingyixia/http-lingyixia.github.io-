---
title: SameAndValid
date: 2019-03-13 14:01:28
catrgory: 深度学习
tags: [CNN]
---
>>记录Same和Valid两种Padding方式卷积或pooling后的大小

#计算公式
$$
(W+padding-K)/stride+1
$$
#Same
Same方式表示当stride为1的时候无论kernel_size是多少,要补充padding的数量需要使前后大小相同,因此,padding的计算方式为:
$$
W+X-K+1=W\\
X=K-1
$$
也就是说需要补充X的padding(X需要平均到两边)
eg:
![](/img/samePadding.png)

#Valid
Valid的方式使不补充padding,当最后不够一个stride的时候直接就不要了.
![](/img/validPadding.png)

#说明
无论哪种方式,计算方式永远不变,只不过Same方式需要先计算padding数量,Valid方式直接padding=0计算即可.