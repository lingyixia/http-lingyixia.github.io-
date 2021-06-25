---
title: SpatialPyramidPooling
date: 2021-06-25 10:27:27
category: 神经网络
tags: [卷积]
---
中文名：空间金字塔卷积，通常卷积之后会跟一个全链接层，这就造成如果输入图片size不同，卷积之后size也不同，输入全链接层的feature就不同，这样压根不能运行，通常的解决方案是通过clip/wrap预先将图片处理成相同size，现在可以用该方法来代替。
<!--more-->
#赘述
通常的CNN结构是这样的：
$$
inputs \rightarrow clip/wrap \rightarrow CNN \rightarrow flat \rightarrow dense
$$

加入ssp后结构是这样的:
$$
inputs \rightarrow CNN \rightarrow ssp \rightarrow dense
$$

#对比
- 原始：
$8 \times 204 \times 196 \times 3 \stackrel{cnn}{\rightarrow}  8 \times 13 \times 14 \times 256 \stackrel{flat}{\rightarrow} 8 \times 46592$
$8 \times 302 \times 197 \times 3 \stackrel{cnn}{\rightarrow}  8 \times 19 \times 14 \times 256  \stackrel{flat}{\rightarrow} 8 \times 68096$
>>显然两者最后的输出不可能输入在同一个dense中训练。

- clip/warp
$8 \times 204 \times 196 \times 3  \stackrel{clip}{\rightarrow} 8 \times 224 \times 224 \times 3  \stackrel{cnn}{\rightarrow} 8 \times 13 \times 13 \times 256  \stackrel{flat}{\rightarrow} 8 \times 43264$
$8 \times 302 \times 197 \times 3  \stackrel{clip}{\rightarrow} 8 \times 224 \times 224 \times 3  \stackrel{cnn}{\rightarrow} 8 \times 13 \times 13 \times 256  \stackrel{flat}{\rightarrow} 8 \times 43264$
- ssp
$8 \times 204 \times 196 \times 3 \stackrel{cnn}{\rightarrow}  8 \times 13 \times 14 \times 256 \stackrel{ssp}{\rightarrow} 8 \times 5376$
$8 \times 302 \times 197 \times 3 \stackrel{cnn}{\rightarrow}  8 \times 19 \times 14 \times 256  \stackrel{ssp}{\rightarrow} 8 \times 5376$

#代码
```
import tensorflow as tf


def SpatialPyramidPooling(previous_conv, out_pool_size_list):
    b, w, h, c = previous_conv.shape
    for index, pool_size in enumerate(out_pool_size_list):
        w_wid = tf.cast(tf.math.ceil(w / pool_size), tf.int64)
        h_wid = tf.cast(tf.math.ceil(h / pool_size), tf.int64)
        max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(w_wid, h_wid), strides=(w_wid, h_wid), padding='same')
        result = tf.reshape(max_pooling(previous_conv), (b, -1))
        spp = result if index == 0 else tf.concat([spp, result], axis=-1)
    return spp


if __name__ == '__main__':
    inputs = tf.random.normal(shape=(8, 19, 14, 256))
    result = SpatialPyramidPooling(inputs, out_pool_size_list=[4, 2, 1])
    print(result.shape)
```