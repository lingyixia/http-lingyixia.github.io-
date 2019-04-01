---
title: BLMO总结
date: 2019-04-01 17:15:53
category: 神经网络
tags: [词向量,预训练]
---
>BLMO是为了解决word2vec的一词多意而诞生的,比如“苹果”这个词,在word2vec中的词向量是确定的,无论语境是在讲水果还是手机,但是在BLMO中单个词的词向量是随上下文而变化的,即动态的词向量,原理就是在将词转化为此向量的时候让其通过一个网络,得到上下文信息,然后在将其与预训练的静态词向量加权获得真正要在下游任务使用的词向量。

#基本框架
![](/img/BLMO1.png)
![](/img/BLMO2.png)
也就是说,在我们想进行一个下游任务比如文本分类,本来需要输入是预训练的词向量,直接将词向量输入模型即可,但是现在我们需要首先将输入词通过BLMO,得到新的词向量,然后在将其输入模型.
#使用方式
[这个教程相当不错](http://www.cnblogs.com/jiangxinyang/p/10235054.html)
#源码部分解析
主要模型都在`bilm/model`中的`BidirectionalLanguageModelGraph`类.

1. word embedding or word_char embedding:
```
 def __init__(self, options, weight_file, ids_placeholder,
                 use_character_inputs=True, embedding_weight_file=None,
                 max_batch_size=128):
```

对应`_build`函数

```

    def _build(self):
        if self.use_character_inputs:
            self._build_word_char_embeddings()
        else:
            self._build_word_embeddings()
        self._build_lstms()
```
>>也就是说无论输入的是字符还是词,`use_character_inputs=True`都默认使用`char_embedding`,一般都这样设置,`_build_word_embeddings`虽然写了,但是似乎就没什么作用(按道理应该是用该函数得到预训练的词向量然后和上下文词向量加权,但是代码中似乎并没有用它).

2. 构建word_char embedding
这部分代码的输出要输入LSTM,应该是总体模型中最复杂的一部分,代码较长,只挑重点部分(函数`_build_word_char_embeddings`):

```
...
 with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                    "char_embed", [n_chars, char_embed_dim],
                    dtype=DTYPE,
                    initializer=tf.random_uniform_initializer(-1.0, 1.0)
            )
            # shape (batch_size, unroll_steps, max_chars, embed_dim)
            self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    self.ids_placeholder)
```
>>这部分是得到`char_embedding`的