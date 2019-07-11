---
layout: post
title: tensorflowFunctions
category: 深度学习
tags: [tensorflow]
date: 2018-12-25 18:17:37
description: tensorflow中常用函数小记
---
tensorflow中常用函数小记(先暂记,以后整理)
<!--more-->
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
#拼接
##tf.concat
>>原型:`tf.concat(values, axis, name='concat')`:按照指定的已经存在的轴进行拼接

eg:略
##tf.stack()
>>原型:`tf.stack(values, axis=0, name='')`:按照指定的新建的轴进行拼接,用于连接tensor，与concat不同的是stack增加维度数量,concat不增加维度数量。

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

#tf.scatter_nd()
给全为0的tensor插入数据，参数说明:
```
indices: shape的坐标
updates: 实际数据，根据indices中的坐标插入shape中
shape: 一个该shape的全为0的tensor
```
eg:(简单例子直接百度)
```
import tensorflow as tf

indices = tf.constant([[0, 1], [2, 3]])
updates = tf.constant([[5, 5, 5, 5],
                       [8, 8, 8, 8]])
shape = tf.constant([4, 4, 4])
scatter = tf.scatter_nd(indices, updates, shape)
with tf.Session() as sess:
    print(sess.run(scatter))
输出:
[[[0 0 0 0]
  [5 5 5 5]
  [0 0 0 0]
  [0 0 0 0]]

 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]

 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  [8 8 8 8]]

 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]]
```

>>解释: `indices`包含两个:[0,1]和[2,3],分别表示`updates`中[5, 5, 5, 5]和[8, 8, 8, 8]在`shape`为(4,4,4)的全为0的tensor中的位置,即[5,5,5,5]的位置是(0,1,*),[8,8,8,8]的位置是(2,3,*)

#判断不合法值
>>nan:not a number,inf:infinity只有这两个!!!其他的:np.NAN和np.NaN就是nan,np.NINF就是-inf

##np.isfinite
>>判断是否是nan或inf

eg:
```
import numpy as np

print(np.isfinite([np.inf, np.log(0), 1, np.nan]))
输出:
/opt/home/chenfeiyu/PythonWorkSpace/MyPointerGeneratorDataset/ttttt.py:9: RuntimeWarning: divide by zero encountered in log
  print(np.isfinite([np.inf, np.log(0), 1, np.nan]))
[False False  True False]
```
注意:此时系统对于$np.log(0)$给出的是warning,但是对于1/0就没办法判断是否是nan或inf了,因为对于分母为零系统给出的是error,程序会直接中断。
##np.isinf
>>判断是否为infinity。

eg:
```
import numpy as np

print(np.isinf([np.inf, np.log(0), 1, np.nan]))
输出:
/opt/home/chenfeiyu/PythonWorkSpace/MyPointerGeneratorDataset/ttttt.py:9: RuntimeWarning: divide by zero encountered in log
  print(np.isinf([np.inf, np.log(0), 1, np.nan]))
[ True  True False False]
```
##np.nan
>>判断是否为not a number

```
import numpy as np

print(np.isnan([np.inf, np.log(0), 1, np.nan]))
输出:
/opt/home/chenfeiyu/PythonWorkSpace/MyPointerGeneratorDataset/ttttt.py:9: RuntimeWarning: divide by zero encountered in log
  print(np.isnan([np.inf, np.log(0), 1, np.nan]))
[False False False  True]
```
#抽取
##tf.slice()
>>原型:tf.slice(inputs,begin,size,name='')从inputs的指定位置**连续**的取,大小为size

```
import tensorflow as tf
inputs = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
data = tf.slice(inputs, [1, 0, 0], [1, 2, 2])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(data.eval())
输出:
[[[3 3]
  [4 4]]]
```
说明:begin和size必和inputs的shape同型,上例中指的是从inputs的[1,0,0]开始,即第一个3处,取第一个维度size为1,第二个维度size为2,第三个维度size为2,即得到上述输出.

##tf.gather()
>>原型:`tf.gather(params, indices, validate_indices=None, name=None)`:按照指定的下标集合从`params`的**axis=0**中抽取子集,适合抽取不连续区域的子集.

```
import tensorflow as tf

params = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
indices=tf.constant([1,2])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.gather(params, indices)))
输出:
[[[3 3 3]
  [4 4 4]]

 [[5 5 5]
  [6 6 6]]]
```
说明:上诉是一般用法,实际的操作为[params[1],params[2]],无论indices是多少维,params直接找最里面的整形数字。

##tf.gather_nd()
>>原型同上,唯一不同点是该函数取得不是**axis=0**,下面的例子阐述了两个函数的区别

```
import tensorflow as tf

params = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
indices = tf.constant([[[0,1],[1,1]]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.gather(params, indices)))
输出:
[[[[[1 1 1]
    [2 2 2]]

   [[3 3 3]
    [4 4 4]]]


  [[[3 3 3]
    [4 4 4]]

   [[3 3 3]
    [4 4 4]]]]]

import tensorflow as tf

params = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
indices = tf.constant([[[0,1],[1,1]]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.gather_nd(params, indices)))
输出:
[[[2 2 2]
  [4 4 4]]]
```
说明:对于indices = tf.constant(\[[\[0,1],[1,1]]]),`gather`函数的操作是\[[\[params[0],params[1]],[params[1],params[1]]]],即直接找整形.而`gather_nd`函数的操作是[\[params[0,1],params[1,1]]],即找整形外面一层.

#tf.split()
>>原型为:`split(value, num_or_size_splits, axis=0, num=None, name="split")`:将value分裂,若`num_or_size_splits`为一个整形,则把`value`的第`axis`维分为`num_or_size_splits`个`Tensor`(此时的`value`的`axis`必须满足整除`num_or_size_splits`),若`num_or_size_splits`为一个一维`Tensor`,则分成该`Tensor`的`shape`个`Tensor`(此时该`Tensor`各个维度相加必须等于该`value`的`axis`)

```
eg1:
import tensorflow as tf

value = tf.Variable(initial_value=tf.truncated_normal(shape=[6, 30]))
split0, split1, split2 = tf.split(value, 3, 0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(split0.shape)
    print(split1.shape)
    print(split2.shape)
输出:
(2, 30)
(2, 30)
(2, 30)

import tensorflow as tf

value = tf.Variable(initial_value=tf.truncated_normal(shape=[6, 30]))
split0, split1, split2 = tf.split(value, [15,4,11], 1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(split0.shape)
    print(split1.shape)
    print(split2.shape)
输出:
(6, 15)
(6, 4)
(6, 11)
```

#损失函数
##sparse_softmax_cross_entropy_with_logits
eg:
```
import tensorflow as tf

if __name__ == '__main__':
    logits=tf.constant(value=[[0.00827806,-0.03050169],[-0.01209893,-0.03642108],[-0.0045999,-0.01193358],[-0.00983661,-0.04756571],[0.00212166,-0.05041311]])
    labels = tf.constant(value=[1,1,1,0,0])
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(losses.eval())
#输出:[0.71272504 0.70538217 0.6968208  0.6744606  0.6672247 ]
```
>>sparse_softmax_cross_entropy_with_logits的作用是首先将logits每一行做`softmax`,然后对labels每一个label转为one_hot,最后计算$\sum_i^n y\prime ln y$(e为底数)

##softmax_cross_entropy_with_logits
eg:
```
import tensorflow as tf

if __name__ == '__main__':
    logits=tf.constant(value=[[0.00827806,-0.03050169],[-0.01209893,-0.03642108],[-0.0045999,-0.01193358],[-0.00983661,-0.04756571],[0.00212166,-0.05041311]])
    labels = tf.constant(value=[[0,1],[0,1],[0,1],[1,0],[1,0]])
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(losses.eval())
#输出:[0.71272504 0.70538217 0.6968208  0.6744606  0.6672247 ]
```
>>注意，下面的函数的labels本来就是one_hot类型,下面函数和上面唯一的不同就是这个,即下面的无需将labels转为one_hot,而是直接输入one_hot类型。