---
layout: post
title: tensorflowFunctions
category: 深度学习
tags: [tensorflow]
date: 2018-12-25 18:17:37
description: tensorflow中常用函数小记
---
>tensorflow中常用函数小记(先暂记,以后整理)

#tf.tile()
用于向量扩张，参数说明:
```
tile(input,multiples,name=None)
input: 一个tensor
multiples: 一个tensor,维度的数量和input维度的数量必然相同
eg:
import tensorflow as tf

raw = tf.Variable(tf.random_normal(shape=(1, 3, 2)))
multi = tf.tile(raw, multiples=[2, 1, 3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(raw.eval())
    print('-----------------------------')
    print(sess.run(multi))

输出:
[[[-1.19627476 -0.66122055]
  [ 1.45084798 -0.87026799]
  [ 0.60792369  0.39918834]]]
-----------------------------
[[[-1.19627476 -0.66122055 -1.19627476 -0.66122055 -1.19627476 -0.66122055]
  [ 1.45084798 -0.87026799  1.45084798 -0.87026799  1.45084798 -0.87026799]
  [ 0.60792369  0.39918834  0.60792369  0.39918834  0.60792369  0.39918834]]

 [[-1.19627476 -0.66122055 -1.19627476 -0.66122055 -1.19627476 -0.66122055]
  [ 1.45084798 -0.87026799  1.45084798 -0.87026799  1.45084798 -0.87026799]
  [ 0.60792369  0.39918834  0.60792369  0.39918834  0.60792369  0.39918834]]]

解释: multiples各个维度表示对input相应维度重复多少倍,如该实例中表示对input的第0个维度重复2次，第1个维度重复1次,第2个维度重复3次，因此得到的shape必然是(2,3,6)
```