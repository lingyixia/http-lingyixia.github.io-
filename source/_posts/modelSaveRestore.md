---
title: CheckPoint的保存和恢复
date: 2019-02-07 14:33:14
category: 深度学习
tags: [tensorflow]
---
>本文用于记录`tensorflow`的`tf.train.Saver()`函数

一个模型保存后会有四个文件:
1. model.ckpt.meta文件: 图结构信息
2. model.ckpt.data-*+model.ckpt.index:保存变量取值
3. checkpoint:保存模型名称,即`restore`的时候的需要传入的名称

#模型保存
```
import tensorflow as tf

#save的路径不需要手动创建
def main(argv=None):
    v1 = tf.Variable(initial_value=tf.constant(value=1.0, shape=[1]), name='v1')
    v2 = tf.Variable(initial_value=tf.constant(value=2.0, shape=[1]), name='v2')
    result = tf.add(x=v1, y=v2, name='add')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.save(sess, './model/model.ckpt')


if __name__ == '__main__':
    tf.app.run()
```

#模型恢复
```
import tensorflow as tf

def main(argv=None):
    v1 = tf.Variable(initial_value=tf.constant(value=1.0, shape=[1]), name='v1')
    v2 = tf.Variable(initial_value=tf.constant(value=2.0, shape=[1]), name='v2')
    result = tf.add(x=v1, y=v2, name='add')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './model/model.ckpt')
        print(result.eval())

if __name__ == '__main__':
    tf.app.run()
```

>>注意回复模型的时候不需要`tf.global_variables_initializer().run()`,这种方式是先把原始图构建出来，然后在回复模型,把模型保存的变量恢复到刚刚构建的图中,还可以不定义原图,直接加载保存的图.

```
import tensorflow as tf

def main(argv=None):
    saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
    with tf.Session() as sess:
        saver.restore(sess, './model/model.ckpt')
        print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))

if __name__ == '__main__':
    tf.app.run()
```
>>在恢复模型的时候,还可以把保存的变量加载到其他变量中。

```
import tensorflow as tf

def main(argv=None):
    u1 = tf.Variable(initial_value=tf.constant(value=1.0, shape=[1]), name='other-v1')
    u2 = tf.Variable(initial_value=tf.constant(value=2.0, shape=[1]), name='other-v2')
    result = tf.add(x=u1, y=u2, name='add')
    saver = tf.train.Saver({'v1': u1, 'v2': u2})
    with tf.Session() as sess:
        saver.restore(sess, './model/model.ckpt')
        print(result.eval())


if __name__ == '__main__':
    tf.app.run()
```
>>`saver = tf.train.Saver({'v1': u1, 'v2': u2})`的作用是把原来名称name为v1的变量现在加载到变量u1(名称name为other-v1)中,这个方式的作用之一是方便使用滑动平均模型。

```
import tensorflow as tf
 
v = tf.Variable(0, dtype=tf.float32, name="v")
for variables in tf.global_variables():
    print(variables.name) # v:0
 
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
for variables in tf.global_variables():
    print(variables.name) # v:0
                          # v/ExponentialMovingAverage:0
 
saver = tf.train.Saver()
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    saver.save(sess, "Model/model_ema.ckpt")
if __name__ == '__main__':
    tf.app.run()
```

```
#获取影子变量方式1
import tensorflow as tf

def main(argv=None):
    v = tf.Variable(0, dtype=tf.float32, name="v")
    saver = tf.train.Saver({"v/ExponentialMovingAverage": v})

    with tf.Session() as sess:
        saver.restore(sess, "./Model/model_ema.ckpt")
        print(sess.run(v))

if __name__ == '__main__':
    tf.app.run()
```

```
#获取影子变量方式2
import tensorflow as tf

def main(argv=None):
    v = tf.Variable(0, dtype=tf.float32, name="v")
    # 注意此处的变量名称name一定要与已保存的变量名称一致
    ema = tf.train.ExponentialMovingAverage(0.99)
    print(ema.variables_to_restore())
    # {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
    # 此处的v取自上面变量v的名称name="v"
    saver = tf.train.Saver(ema.variables_to_restore())
    with tf.Session() as sess:
        saver.restore(sess, "./Model/model_ema.ckpt")
        print(sess.run(v))
if __name__ == '__main__':
    tf.app.run()
```

#常量保存
```
import tensorflow as tf
from tensorflow.python.framework import graph_util

def main(argv=None):
    v1 = tf.Variable(initial_value=tf.constant(value=1.0, shape=[1]), name='v1')
    v2 = tf.Variable(initial_value=tf.constant(value=2.0, shape=[1]), name='v2')
    result = tf.add(x=v1, y=v2, name='add')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess=sess, input_graph_def=graph_def,
                                                                     output_node_names=['add'])
        with tf.gfile.GFile(name='./model/combined.pd', mode='wb') as f:
            f.write(output_graph_def.SerializeToString())
if __name__ == '__main__':
    tf.app.run()
```

#常量恢复
```
import tensorflow as tf
from tensorflow.python.platform import gfile

def main(argv=None):
    with tf.Session() as sess:
        model_filename = './model/combined.pd'
        with gfile.FastGFile(model_filename, mode='rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        result = tf.import_graph_def(graph_def, return_elements=['add:0'])
        print(sess.run(result))

if __name__ == '__main__':
    tf.app.run()
```