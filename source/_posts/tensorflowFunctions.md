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
#tf.stack()
用于连接tensor，与concat不同的是stack增加维度数量，concat不增加维度数量。
```
eg 1:
import tensorflow as tf

a = tf.constant([[1,1,1],[2,2,2],[3,3,3],[4,4,4]]) # shape (4,3)
b = tf.constant([[5,5,5],[6,6,6],[7,7,7],[8,8,8]]) # shape (4,3)
ab = tf.stack([a,b], axis=0)

with tf.Session() as sess:
    print(ab.shape)
    print(sess.run(ab))

输出:
(2, 4, 3)
[[[1 1 1]
  [2 2 2]
  [3 3 3]
  [4 4 4]]

 [[5 5 5]
  [6 6 6]
  [7 7 7]
  [8 8 8]]]

eg 2:
import tensorflow as tf

a = tf.constant([[1,1,1],[2,2,2],[3,3,3],[4,4,4]]) # shape (4,3)
b = tf.constant([[5,5,5],[6,6,6],[7,7,7],[8,8,8]]) # shape (4,3)
ab = tf.stack([a,b], axis=1)

with tf.Session() as sess:
    print(ab.shape)
    print(sess.run(ab))
输出:
(4, 2, 3)
[[[1 1 1]
  [5 5 5]]

 [[2 2 2]
  [6 6 6]]

 [[3 3 3]
  [7 7 7]]

 [[4 4 4]
  [8 8 8]]]

eg 3:
import tensorflow as tf

a = tf.constant([[1,1,1],[2,2,2],[3,3,3],[4,4,4]]) # shape (4,3)
b = tf.constant([[5,5,5],[6,6,6],[7,7,7],[8,8,8]]) # shape (4,3)
ab = tf.stack([a,b], axis=2)

with tf.Session() as sess:
    print(ab.shape)
    print(sess.run(ab))
输出:
(4, 3, 2)
[[[1 5]
  [1 5]
  [1 5]]

 [[2 6]
  [2 6]
  [2 6]]

 [[3 7]
  [3 7]
  [3 7]]

 [[4 8]
  [4 8]
  [4 8]]]
```
>>也就是说把两个tensor相应维度合并成一个新维度，如eg2,把第1维度的[1,1,1]和[5,5,5]合并为\[[1,1,1],[5,5,5]],同理tf.unstack().
与conact不同的一点是conact中tensor维度有多少axis就最大是多少,stack()的axis可以大1个维度。