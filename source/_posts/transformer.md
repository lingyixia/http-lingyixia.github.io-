---
title: Transform
date: 2019-04-05 13:31:37
category: 深度学习
tags: [Attention]
---
本文章只用于自己记录,详细内容好多博客已经讲的很清楚了。
[参考博客1](https://jalammar.github.io/illustrated-transformer/)
[参考博客2](https://zhuanlan.zhihu.com/p/47282410?utm_source=wechat_session&utm_medium=social&s_r=0)
[参考博客3](https://blog.csdn.net/yiyele/article/details/81913031)
[参考平博客(最佳)](https://medium.com/@mromerocalvo/dissecting-bert-part1-6dcf5360b07f)
[残差网络](https://lingyixia.github.io/2019/05/01/CNNdevelopment/#ResNet)
[我的实现](https://github.com/lingyixia/Attention)
<!--more-->
#整体架构
![](/img/transform2.jpg)
解释1:左半部分是`encoder`,方框中是一个`encoder cell`,右半部分是`decoder`,方框中是一个`decoder cell`。
解释2:一个`encoder cell`包含四层:`self-Multi-Attention`+`ResNet And Norm`+`Feed-Forward`+`ResNet And Norm`,名字和图中不太一样,按照层数数下即可.
解释3: 一个`decoder cell`包含六层:`Mask-self-Multi-Attention`+`ResNet And Norm`+`encoder-decoder-Multi-Attention`+`ResNet And Norm`+`Feed-Forward`+`ResNet And Norm`,名字和图中不太一样,按照层数数下即可.
解释4:`encoder`阶段的`self-Multi-Attention`和`decoder`阶段的`Mask-self-Multi-Attention`,`encoder-decoder-Multi-Attention`是同一段代码。
解释5:以翻译任务为例,假设当前需要翻译t位置的词汇,`decoder`阶段的`mask-self-Attention`是对0~t-1,即对已经翻译出来部分的attention,故需要做`mask`,防止attention未翻译部分,`encoder-decoder-Attention`是对原文所有的attention。
解释6:`encoder-decoder-Multi-Attention`并不是`self-Attention`,因为它的`Q`是原文状态,`K`和`V`是译文状态
#self-Attention
先上图:
![](/img/transform1.gif)
![](/img/selfattention.jpg)
其实就是计算句子中每个单词对其他所有单词的关注程度:
$$
Attention(Q,K,V)=\frac{softmax(Q \times K^T)}{\sqrt{d_k}} \times V
$$
这里都讲烂了,就不记录了.
#ResNet And Norm
残差网络的作用见[残差网络](https://www.jianshu.com/p/e58437f39f65),Norm的作用自然是加快收敛,防止梯度消失.
#Feed-Forward
这里其实没有什么特殊的东西,源码中用了两个`conv1`,先把特征维度放大,在把特征维度缩小,其实也就是特征提取的作用.
#Positional Encoding
由于在attention的时候仅仅计算了当前词与其他词的相关度,但是并没有其他词的位置信息,试想,输入一个待翻译的句子1,然后将该句中任意两个单词互换位置形成句子2,在当前结构中其实两个句子并没有什么不同,因此需要对每个单词引如其位置信息.在论文中使用的是这样的方式:
$$
PE_{(pos,2i)}=sin(pos/10000^{2i/d_{modle}}) \\
PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{modle}})
$$
其中pos是当前单词在该句子中的位置,比如句子长20,则pos可以是1,2...20.i是在当前单词的第i个维度,比如每个单词有512个维度,则i可以是1,2...512.$d_{modle}$是单词维度,当前例子中即是512.
#其他
1.每一个`self-Attention`都维护一个自己的$W_Q$,$W_K$,$W_V$,也就是生成`Q`,`K`,`V`的全连接神经网络参数,即每个`cell`的这三个值是不同的.
2.在`encoder`阶段的最后一个`encoder cell`会将生成的`K`和`V`传递给`decoder`阶段每个`decoder cell`的`encoder-decoder-Multi-Attention`使用.而`encoder-decoder-Multi-Attention`使用的`Q`是`Mask-self-Multi-Attention`输出的.
3.`decoder`阶段`Mask-self-Multi-Attention`用来Attention翻译过的句子,`encoder-decoder-Multi-Attention`用来Attention原文.
4. 在代码中,的多头 `Multi-Attention` 的实现实际上每个头把维度给均分了,并不是每个头都 attention 所有的维度。
```
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights
```